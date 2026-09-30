#!/usr/bin/env python3
"""
capture_node.py

Exposes ~/sample_position (handeye_calibration_interfaces/srv/SamplePosition).

Each call captures the current camera frame + gripper pose (looked up via TF)
and appends it as a new entry to {request.dataset_dir}/dataset.yaml, writing
the image to {request.dataset_dir}/images/. The dataset's camera_matrix,
dist_coeffs and board config are populated on the first sample for a given
dataset_dir.

Assumes the robot is already stationary at the desired pose when this service
is called -- moving it there is run_points_node's job (~/move_to_next_point).
"""

import pathlib
import time

import numpy as np
import cv2
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
import tf_transformations as tft

from handeye_calibration_interfaces.srv import SamplePosition
from handeye_calibration.utils import imgmsg_to_cv2


class CaptureNode(Node):
    def __init__(self):
        super().__init__('capture_node')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('gripper_frame', 'gripper_link')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')

        # -- board config, passed through into dataset.yaml for calibration_node --
        self.declare_parameter('squares_x', 11)
        self.declare_parameter('squares_y', 8)
        self.declare_parameter('square_length_m', 0.015)
        self.declare_parameter('marker_length_m', 0.011)
        self.declare_parameter('dictionary', '4X4_50')
        self.declare_parameter('legacy_pattern', True)
        self.declare_parameter('min_charuco_corners', 6)

        self.base_frame = self.get_parameter('base_frame').value
        self.gripper_frame = self.get_parameter('gripper_frame').value

        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_bgr = None
        self._image_time = 0.0
        # Debug bookkeeping: how many frames have arrived, and when the newest
        # one landed on the wall clock. Together these show whether _image_cb
        # was starved while a service call was running.
        self._image_count = 0
        self._image_recv_monotonic = None

        self.caminfo_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self._caminfo_cb, 10)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_service(SamplePosition, '~/sample_position', self._srv_sample_position)

        self.get_logger().info('capture_node ready. Service: ~/sample_position')

    # ------------------------------------------------------------
    # Camera / TF helpers
    # ------------------------------------------------------------
    def _caminfo_cb(self, msg):
        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if k[0, 0] == 0.0 or k[1, 1] == 0.0:
            # A webcam has no factory intrinsics: until camera_info_url points at
            # a calibrated file, usb_cam publishes an all-zero CameraInfo. Keep
            # the subscription alive so a later valid message still lands.
            self.get_logger().warn(
                'CameraInfo has zero focal length -- camera is uncalibrated. '
                'Run intrinsics.launch.py first.', once=True)
            return
        self.camera_matrix = k
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.caminfo_sub.destroy()

    def _image_cb(self, msg):
        self.latest_bgr = imgmsg_to_cv2(msg)
        self._image_time = msg.header.stamp
        self._image_count += 1
        self._image_recv_monotonic = time.monotonic()
        # Throttled: the point is to see the stream stop and restart around a
        # blocking call, not to log every frame at 10 Hz.
        self.get_logger().debug(
            f'    [image_cb] frame #{self._image_count} '
            f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}')

    def has_camera_intrinsics(self):
        """camera_matrix is set by _caminfo_cb, processed by the same
        spin() loop during the idle time between service calls -- no
        waiting here, this is a single check against whatever's
        already arrived."""
        return self.camera_matrix is not None

    def get_base_to_gripper(self):
        """Pose of gripper_frame expressed in base_frame, as (R, t),
        from whatever the TF buffer already has -- single attempt, no
        retrying."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.gripper_frame, rclpy.time.Time())
            q, t = tf.transform.rotation, tf.transform.translation
            R = tft.transforms3d.quaternions.quat2mat([q.w, q.x, q.y, q.z])
            # rclpy.time.Time() asks for "latest available", so this stamp is
            # whatever TF has right now -- NOT the image's stamp. Logging it is
            # what makes the pairing mismatch visible.
            self._last_tf_stamp = tf.header.stamp
            self.get_logger().info(
                f'    [tf] {self.base_frame} -> {self.gripper_frame} '
                f'stamp={tf.header.stamp.sec}.{tf.header.stamp.nanosec:09d} '
                f'(requested: latest available) t=[{t.x:+.4f} {t.y:+.4f} {t.z:+.4f}]')
            return R, np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    def grab_latest_image(self):
        """Returns whatever _image_cb last stored (or None if no image
        has arrived yet) -- single check, no waiting for a new frame."""
        if self.latest_bgr is None:
            return None, None
        age = (time.monotonic() - self._image_recv_monotonic
               if self._image_recv_monotonic is not None else float('nan'))
        self.get_logger().info(
            f'    [image] reusing buffered frame #{self._image_count} '
            f'stamp={self._image_time.sec}.{self._image_time.nanosec:09d} '
            f'arrived {age:.3f}s ago (no wait for a fresh frame)')
        if age > 0.5:
            self.get_logger().warn(
                f'    [image] STALE by {age:.3f}s -- this frame predates the '
                'current pose; image/TF pairing is suspect')
        return self.latest_bgr.copy(), self._image_time

    # ------------------------------------------------------------
    def _load_or_init_dataset(self, dataset_path):
        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as fp:
                return yaml.safe_load(fp)

        p = self.get_parameter
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'board': {
                'squares_x': p('squares_x').value,
                'squares_y': p('squares_y').value,
                'square_length_m': p('square_length_m').value,
                'marker_length_m': p('marker_length_m').value,
                'dictionary': p('dictionary').value,
                'legacy_pattern': p('legacy_pattern').value,
                'min_charuco_corners': p('min_charuco_corners').value,
            },
            'entries': [],
        }

    # ------------------------------------------------------------
    def _srv_sample_position(self, request, response):
        srv_t0 = time.monotonic()
        frames_before = self._image_count
        self._last_tf_stamp = None
        self.get_logger().info(
            f'>>> sample_position ENTER (frames received so far: {frames_before})')
        out_dir = pathlib.Path(request.dataset_dir).expanduser()
        images_dir = out_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = out_dir / 'dataset.yaml'

        if not self.has_camera_intrinsics():
            response.success = False
            response.message = ('No valid camera intrinsics received yet '
                                '(zero focal length means uncalibrated).')
            self.get_logger().error(response.message)
            self.get_logger().info(
                f'<<< sample_position EXIT (took {time.monotonic() - srv_t0:.3f}s) '
                '-- no intrinsics')
            return response

        dataset = self._load_or_init_dataset(dataset_path)

        bgr, _ = self.grab_latest_image()
        if bgr is None:
            response.success = False
            response.message = 'No camera frame received yet.'
            self.get_logger().warn(response.message)
            self.get_logger().info(
                f'<<< sample_position EXIT (took {time.monotonic() - srv_t0:.3f}s) '
                '-- no frame')
            return response

        pose = self.get_base_to_gripper()
        if pose is None:
            response.success = False
            response.message = 'TF lookup failed.'
            self.get_logger().warn(response.message)
            self.get_logger().info(
                f'<<< sample_position EXIT (took {time.monotonic() - srv_t0:.3f}s) '
                '-- TF lookup failed')
            return response
        R_gripper2base, t_gripper2base = pose

        # The heart of the matter: image stamp vs TF stamp. For a correctly
        # synchronised sample these are within a frame period of each other.
        # A large positive skew means the image is older than the pose -- the
        # frame belongs to the PREVIOUS robot position.
        # The heart of the matter: image stamp vs TF stamp. For a correctly
        # synchronised sample these are within a frame period of each other.
        # A large skew means the two describe different instants.
        if self._last_tf_stamp is not None:
            img_t = self._image_time.sec + self._image_time.nanosec * 1e-9
            tf_t = self._last_tf_stamp.sec + self._last_tf_stamp.nanosec * 1e-9
            skew = tf_t - img_t
            self.get_logger().info(
                f'    [sync] image stamp {img_t:.3f} | tf stamp {tf_t:.3f} | '
                f'tf - image = {skew:+.3f}s')
            if abs(skew) > 0.2:
                self.get_logger().error(
                    f'    [sync] MISMATCH {skew:+.3f}s -- image and pose describe '
                    'different instants. This sample pairs a frame from one robot '
                    'position with a pose from another.')

        idx = len(dataset['entries'])
        image_file = f'sample_{idx:03d}.png'
        cv2.imwrite(str(images_dir / image_file), bgr)

        dataset['entries'].append({
            'image_file': image_file,
            'R_gripper2base': R_gripper2base.tolist(),
            't_gripper2base': t_gripper2base.tolist(),
        })

        with open(dataset_path, 'w', encoding='utf-8') as fp:
            yaml.safe_dump(dataset, fp, sort_keys=False)

        self.get_logger().info(f'Saved sample {idx} -> images/{image_file}')

        response.success = True
        response.message = f'Saved sample {idx}'
        response.image_file = image_file
        response.sample_index = idx
        self.get_logger().info(
            f'<<< sample_position EXIT (took {time.monotonic() - srv_t0:.3f}s, '
            f'{self._image_count - frames_before} frames arrived during the call)')
        return response


def main():
    rclpy.init()
    node = CaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
