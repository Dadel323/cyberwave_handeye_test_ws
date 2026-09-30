#!/usr/bin/env python3
"""
viewer_node.py

Continuous live view of the calibration camera, taken from the compressed
image transport (`<image_topic>/compressed`, sensor_msgs/CompressedImage).

Subscribing to the compressed stream rather than image_raw keeps a viewer off
the hot path: image_raw at 1280x720 bgr8 is ~2.8 MB per frame, and every extra
subscriber costs another copy of that through the middleware. The compressed
topic is roughly a tenth of it, and JPEG artifacts in a monitoring window do
not matter -- nothing here feeds a solve. (The detection nodes deliberately
stay on image_raw for that reason.)

cv2.imdecode does the decoding, not cv_bridge -- same reason as
utils.imgmsg_to_cv2: cv_bridge's compiled extension is linked against
NumPy 1.x's ABI and crashes under NumPy 2.x.

The window needs a display, so run it from a session that has one (not over a
plain ssh without X forwarding). Press q or Esc in the window to quit.

    ros2 run handeye_calibration viewer_node
    ros2 run handeye_calibration viewer_node --ros-args -p image_topic:=/camera/color/image_raw
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import CompressedImage


class ViewerNode(Node):
    def __init__(self):
        super().__init__('viewer_node')

        # The base topic, matching the `image_topic` parameter every other node
        # in this package takes; '/compressed' is appended here so switching
        # cameras means changing the same value in both places.
        self.declare_parameter('image_topic', '/webcam/image_raw')
        self.declare_parameter('window_name', 'camera')
        self.declare_parameter('show_fps', True)

        base_topic = self.get_parameter('image_topic').value
        self.topic = base_topic.rstrip('/') + '/compressed'
        self.window_name = self.get_parameter('window_name').value
        self.show_fps = self.get_parameter('show_fps').value

        # Sensor-data QoS (best effort, depth 5): a viewer that falls behind
        # should drop frames rather than queue stale ones, and image_transport
        # publishers offer best-effort -- a RELIABLE subscription would simply
        # never match.
        self.create_subscription(
            CompressedImage, self.topic, self._image_cb,
            QoSPresetProfiles.SENSOR_DATA.value)

        # Drives the GUI event loop and the quit key. cv2.imshow is called from
        # here rather than from the image callback so the window stays
        # responsive even when no frames are arriving.
        self.latest_bgr = None
        self.frame_count = 0
        self.fps = 0.0
        self._fps_last_count = 0
        self._fps_last_stamp = self.get_clock().now()
        self.create_timer(0.03, self._gui_tick)
        self.create_timer(1.0, self._fps_tick)

        self.warned_no_frames = False
        self.get_logger().info(f'viewer_node showing {self.topic}. '
                                'Press q or Esc in the window to quit.')

    def _image_cb(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(
                f'Could not decode a "{msg.format}" frame -- skipping it.')
            return
        self.latest_bgr = bgr
        self.frame_count += 1

    def _fps_tick(self):
        """Measured over wall-clock so a stalled stream reads as 0, not as the
        last good rate."""
        now = self.get_clock().now()
        elapsed = (now - self._fps_last_stamp).nanoseconds / 1e9
        if elapsed > 0:
            self.fps = (self.frame_count - self._fps_last_count) / elapsed
        self._fps_last_count = self.frame_count
        self._fps_last_stamp = now

        if self.frame_count == 0 and not self.warned_no_frames:
            self.get_logger().warn(
                f'No frames on {self.topic} yet. Is the camera up, and is '
                'compressed_image_transport installed alongside it?')
            self.warned_no_frames = True

    def _gui_tick(self):
        if self.latest_bgr is None:
            return

        frame = self.latest_bgr
        if self.show_fps:
            frame = frame.copy()
            cv2.putText(frame, f'{self.fps:4.1f} fps  {frame.shape[1]}x{frame.shape[0]}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            self.get_logger().info('Quit key pressed -- shutting down.')
            raise SystemExit(0)


def main():
    rclpy.init()
    node = ViewerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
