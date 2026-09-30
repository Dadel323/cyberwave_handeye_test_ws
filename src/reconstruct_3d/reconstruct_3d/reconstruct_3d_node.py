#!/usr/bin/env python3
import sys
import pathlib

import yaml

import rclpy
from rclpy.node import Node

from handeye_calibration_interfaces.srv import MoveToNextPoint, SampleScene
from reconstruct_3d.utils import reconstruct_point_cloud


class Reconstruct3DOrchestrator(Node):
    """Drives the 3D reconstruction pipeline for a preset number of samples:
    for each sample, calls a service to move to the next point, then a
    service to sample (capture) the current scene. Once all samples are
    collected, reconstructs a point cloud from the resulting dataset."""

    def __init__(self):
        super().__init__('reconstruct_3d_node')

        self.declare_parameter('output_dir', '3d_reconstruction_dataset')
        self.declare_parameter('num_samples', 30)
        self.declare_parameter('point_sampler_node', '/run_points_node')
        self.declare_parameter('scene_capture_node', '/scene_capture_node')

        # -- reconstruction parameters --
        self.declare_parameter('orb_features', 4000)
        self.declare_parameter('match_ratio_test', 0.75)
        self.declare_parameter('min_matches_per_pair', 8)
        self.declare_parameter('max_point_distance_m', 2.0)  # sanity filter

        mover_ns = self.get_parameter('point_sampler_node').value
        capture_ns = self.get_parameter('scene_capture_node').value

        self._move_client = self.create_client(MoveToNextPoint, f'{mover_ns}/move_to_next_point')
        self._sample_client = self.create_client(SampleScene, f'{capture_ns}/sample_scene')

    # ------------------------------------------------------------
    def collect_dataset(self, output_dir, num_samples):
        self.get_logger().info('Waiting for move_to_next_point service...')
        self._move_client.wait_for_service()
        self.get_logger().info('Waiting for sample_scene service...')
        self._sample_client.wait_for_service()

        num_captured = 0
        for i in range(1, num_samples + 1):
            self.get_logger().info(f'[{i}/{num_samples}] Calling move_to_next_point...')
            move_future = self._move_client.call_async(MoveToNextPoint.Request())
            rclpy.spin_until_future_complete(self, move_future)
            move_response = move_future.result()

            if move_response is None:
                self.get_logger().error(f'Sample {i}: move_to_next_point call failed (no response).')
                continue
            if not move_response.success:
                self.get_logger().error(f'Sample {i}: {move_response.message}')
                continue

            self.get_logger().info(f'[{i}/{num_samples}] Calling sample_scene...')
            sample_request = SampleScene.Request()
            sample_request.dataset_dir = output_dir
            sample_future = self._sample_client.call_async(sample_request)
            rclpy.spin_until_future_complete(self, sample_future)
            sample_response = sample_future.result()

            if sample_response is None:
                self.get_logger().error(f'Sample {i}: sample_scene call failed (no response).')
                continue
            if not sample_response.success:
                self.get_logger().error(f'Sample {i}: {sample_response.message}')
                continue

            num_captured += 1
            self.get_logger().info(f'Sample {i}: {sample_response.message}')

        self.get_logger().info(
            f'Dataset collection finished: {num_captured}/{num_samples} usable samples, '
            f'dataset written to {output_dir}')
        return num_captured > 0, num_captured

    # ------------------------------------------------------------
    def run(self):
        p = self.get_parameter
        output_dir = p('output_dir').value
        success, num_captured = self.collect_dataset(output_dir, p('num_samples').value)

        if not success:
            self.get_logger().error('Dataset collection failed (0 samples captured).')
            return False
        self.get_logger().info(f'Captured {num_captured} viewpoints to {output_dir}.')

        manifest_path = pathlib.Path(output_dir).expanduser() / 'manifest.yaml'
        with open(manifest_path, 'r', encoding='utf-8') as fp:
            manifest = yaml.safe_load(fp)

        ply_path = reconstruct_point_cloud(
            output_dir, manifest,
            n_features=p('orb_features').value,
            ratio_test=p('match_ratio_test').value,
            min_matches=p('min_matches_per_pair').value,
            max_dist=p('max_point_distance_m').value,
            out_dir=output_dir,
        )
        if ply_path is None:
            self.get_logger().error('3D reconstruction failed.')
            return False

        self.get_logger().info(f'Point cloud written to {ply_path}')
        return True


def main():
    rclpy.init()
    node = Reconstruct3DOrchestrator()
    success = node.run()
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
