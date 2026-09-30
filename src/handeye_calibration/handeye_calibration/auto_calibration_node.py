import sys

import time
import rclpy
from rclpy.node import Node

from handeye_calibration_interfaces.srv import CalibrateDataset, SamplePosition, MoveToNextPoint


class CalibrationOrchestrator(Node):
    """Drives the calibration pipeline for a preset number of samples.

    Per sample: call point_sampler_node's ~/move_to_next_point to reach the
    next pose, then capture_node's ~/sample_position to record the frame and
    the TF lookup. Once all samples are collected, call calibrate_dataset to
    solve hand-eye over the result.

    The sampler is whatever exposes ~/move_to_next_point -- run_points_node
    replaying a recorded manifest (the openarm setup), or motor_bridge
    prompting for each pose by hand (the SO-101 setup)."""

    def __init__(self):
        super().__init__('auto_calibration_node')

        self.declare_parameter('output_dir', 'handeye_dataset')
        self.declare_parameter('calibration_method', 'DANIILIDIS')
        self.declare_parameter('num_samples', 200)
        self.declare_parameter('point_sampler_node', '/run_points_node')
        self.declare_parameter('capture_node', '/capture_node')

        capture_ns = self.get_parameter('capture_node').value

        self._sample_client = self.create_client(SamplePosition, f'{capture_ns}/sample_position')
        self._calibrate_client = self.create_client(CalibrateDataset, 'calibrate_dataset')

        mover_ns = self.get_parameter('point_sampler_node').value
        self._move_client = self.create_client(
            MoveToNextPoint, f'{mover_ns}/move_to_next_point')

    # ------------------------------------------------------------
    def _call(self, client, request):
        """Call a service and return its response, or None."""
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def collect_dataset(self, output_dir, num_samples):
        self.get_logger().info('Waiting for move_to_next_point service...')
        self._move_client.wait_for_service()
        self.get_logger().info('Waiting for sample_position service...')
        self._sample_client.wait_for_service()

        num_captured = 0
        for i in range(1, num_samples + 1):
            loop_t0 = time.monotonic()
            self.get_logger().info(
                f'===== SAMPLE {i}/{num_samples} BEGIN =====')
            self.get_logger().info(f'[{i}/{num_samples}] Calling move_to_next_point...')
            move_t0 = time.monotonic()
            move_response = self._call(self._move_client, MoveToNextPoint.Request())
            move_dt = time.monotonic() - move_t0
            self.get_logger().info(
                f'[{i}/{num_samples}] move_to_next_point returned after {move_dt:.3f}s')

            if move_response is None:
                self.get_logger().error(f'Sample {i}: move_to_next_point call failed (no response).')
                continue
            if not move_response.success:
                self.get_logger().error(f'Sample {i}: {move_response.message}')
                continue

            # No settle wait here: sample_position is called the instant the
            # move service returns. This gap is logged because it is exactly
            # where a stale frame gets paired with a fresh pose.
            gap = time.monotonic() - move_t0 - move_dt
            self.get_logger().info(
                f'[{i}/{num_samples}] Calling sample_position '
                f'({gap * 1000:.1f}ms after move returned, no settle wait)...')
            sample_request = SamplePosition.Request()
            sample_request.dataset_dir = output_dir
            sample_t0 = time.monotonic()
            sample_response = self._call(self._sample_client, sample_request)
            self.get_logger().info(
                f'[{i}/{num_samples}] sample_position returned after '
                f'{time.monotonic() - sample_t0:.3f}s')

            if sample_response is None:
                self.get_logger().error(f'Sample {i}: sample_position call failed (no response).')
                continue
            if not sample_response.success:
                self.get_logger().error(f'Sample {i}: {sample_response.message}')
                continue

            num_captured += 1
            self.get_logger().info(f'Sample {i}: {sample_response.message}')
            self.get_logger().info(
                f'===== SAMPLE {i}/{num_samples} END '
                f'(total {time.monotonic() - loop_t0:.3f}s) =====')

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

        self.get_logger().info('Waiting for calibrate_dataset service...')
        self._calibrate_client.wait_for_service()

        calib_request = CalibrateDataset.Request()
        calib_request.dataset_dir = output_dir
        calib_request.method = p('calibration_method').value

        self.get_logger().info('Calling CalibrateDataset...')
        future = self._calibrate_client.call_async(calib_request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if not response.success:
            self.get_logger().error('Calibration solve failed.')
            return False

        self.get_logger().info(f'Calibration complete. Result written to {response.result_yaml_path}')
        self.get_logger().info(f'  translation: {response.translation}')
        self.get_logger().info(f'  rotation_xyzw: {response.rotation_xyzw}')
        self.get_logger().info(f'  leave-one-out stddev: {response.stability_stddev}')
        return True


def main():
    rclpy.init()
    node = CalibrationOrchestrator()
    success = node.run()
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
