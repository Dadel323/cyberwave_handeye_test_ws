#!/usr/bin/env python3
"""
scene_capture_node.py

Exposes ~/sample_scene (handeye_calibration_interfaces/srv/SampleScene).

Each call captures the current color + depth + point cloud frame plus the
camera's pose (looked up via TF, in base_frame) and appends it as a new
entry to {request.dataset_dir}/manifest.yaml, writing image/depth/cloud
files to {request.dataset_dir}/images/. The manifest's camera_matrix and
dist_coeffs are populated on the first sample for a given dataset_dir.

Assumes the robot is already stationary at the desired pose when this
service is called -- moving it there is run_points_node's (or dome_mover's)
job (~/move_to_next_point).
"""

import pathlib

import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
import tf_transformations as tft

from handeye_calibration_interfaces.srv import SampleScene
from handeye_calibration.utils import imgmsg_to_cv2


class SceneCaptureNode(Node):
    def __init__(self):
        super().__init__('scene_capture_node')

        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('pointcloud_topic', '/camera/depth/points')

        self.camera_frame = self.get_parameter('camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_bgr = None
        self.latest_depth = None
        self.latest_cloud_msg = None

        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self._caminfo_cb, 10)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_cb, 10)
        self.create_subscription(
            Image, self.get_parameter('depth_image_topic').value, self._depth_cb, 10)
        self.create_subscription(
            PointCloud2, self.get_parameter('pointcloud_topic').value, self._cloud_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_service(SampleScene, '~/sample_scene', self._srv_sample_scene)

        self.get_logger().info('scene_capture_node ready. Service: ~/sample_scene')

    # ------------------------------------------------------------
    # Camera / TF helpers
    # ------------------------------------------------------------
    def _caminfo_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _image_cb(self, msg):
        self.latest_bgr = imgmsg_to_cv2(msg)

    def _depth_cb(self, msg):
        self.latest_depth = imgmsg_to_cv2(msg)

    def _cloud_cb(self, msg):
        self.latest_cloud_msg = msg

    def get_camera_in_base_frame(self):
        """Pose of camera_frame expressed in base_frame, as (R, t), from
        whatever the TF buffer already has -- single attempt, no retrying."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None
        q = tf_msg.transform.rotation
        t = tf_msg.transform.translation
        R = tft.transforms3d.quaternions.quat2mat([q.w, q.x, q.y, q.z])
        return R, np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------
    @staticmethod
    def _save_depth(depth, images_dir, idx):
        """16-bit integer depth (e.g. mm) saves losslessly as PNG;
        float depth (e.g. metres) can't be stored losslessly in PNG,
        so it's saved as .npy instead."""
        if np.issubdtype(depth.dtype, np.integer):
            depth_file = f'sample_{idx:03d}_depth.png'
            cv2.imwrite(str(images_dir / depth_file), depth)
        else:
            depth_file = f'sample_{idx:03d}_depth.npy'
            np.save(str(images_dir / depth_file), depth.astype(np.float32))
        return depth_file

    @staticmethod
    def _save_cloud(cloud_msg, images_dir, idx):
        """Extracts available fields (x, y, z, and rgb if present) from
        the PointCloud2 message and saves as a plain (N, k) float array."""
        field_names = [f.name for f in cloud_msg.fields]
        want = [n for n in ('x', 'y', 'z') if n in field_names]
        if 'rgb' in field_names:
            want.append('rgb')
        points = pc2.read_points_numpy(cloud_msg, field_names=want, skip_nans=True)
        cloud_file = f'sample_{idx:03d}_cloud.npy'
        np.save(str(images_dir / cloud_file), points)
        return cloud_file

    # ------------------------------------------------------------
    def _load_or_init_manifest(self, manifest_path):
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as fp:
                return yaml.safe_load(fp)

        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'entries': [],
        }

    # ------------------------------------------------------------
    def _srv_sample_scene(self, request, response):
        out_dir = pathlib.Path(request.dataset_dir).expanduser()
        images_dir = out_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / 'manifest.yaml'

        if self.camera_matrix is None:
            response.success = False
            response.message = 'No camera intrinsics received yet.'
            self.get_logger().error(response.message)
            return response

        if self.latest_bgr is None or self.latest_depth is None or self.latest_cloud_msg is None:
            response.success = False
            response.message = 'No color/depth/cloud frame received yet.'
            self.get_logger().warn(response.message)
            return response

        pose = self.get_camera_in_base_frame()
        if pose is None:
            response.success = False
            response.message = 'TF lookup failed.'
            self.get_logger().warn(response.message)
            return response
        R_camera2base, t_camera2base = pose

        manifest = self._load_or_init_manifest(manifest_path)
        idx = len(manifest['entries'])

        image_file = f'sample_{idx:03d}.png'
        cv2.imwrite(str(images_dir / image_file), self.latest_bgr)
        depth_file = self._save_depth(self.latest_depth, images_dir, idx)
        cloud_file = self._save_cloud(self.latest_cloud_msg, images_dir, idx)

        manifest['entries'].append({
            'image_file': image_file,
            'depth_file': depth_file,
            'cloud_file': cloud_file,
            'R_camera2base': R_camera2base.tolist(),
            't_camera2base': t_camera2base.tolist(),
        })

        with open(manifest_path, 'w', encoding='utf-8') as fp:
            yaml.safe_dump(manifest, fp, sort_keys=False)

        self.get_logger().info(f'Saved sample {idx} -> images/{image_file}, {depth_file}, {cloud_file}')

        response.success = True
        response.message = f'Saved sample {idx}'
        response.image_file = image_file
        response.sample_index = idx
        return response


def main():
    rclpy.init()
    node = SceneCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
