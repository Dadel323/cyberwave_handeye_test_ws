#!/usr/bin/env python3
"""
intrinsics_node.py

Monocular intrinsics calibration for a plain webcam, using the same ChArUco
board the hand-eye pipeline already uses.

A webcam publishes an all-zero CameraInfo (unlike a depth camera, which ships
factory intrinsics), so capture_node has nothing usable to write into
dataset.yaml. This node produces those intrinsics.

The board is built via handeye_calibration.utils.build_board -- the same
function capture_node/calibration_node use -- so the board model (including
legacy_pattern) is identical across intrinsics and the hand-eye solve.

Services:
  ~/capture    (std_srvs/Trigger) -- detect the board in the current frame and
                                     store the correspondences as one view
  ~/calibrate  (std_srvs/Trigger) -- solve over all stored views and write the
                                     camera_info YAML
  ~/reset      (std_srvs/Trigger) -- drop all stored views
"""

import pathlib

import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from handeye_calibration.utils import build_board, imgmsg_to_cv2


class IntrinsicsNode(Node):
    def __init__(self):
        super().__init__('intrinsics_node')

        self.declare_parameter('image_topic', '/webcam/image_raw')
        self.declare_parameter('output_path', '')
        self.declare_parameter('camera_name', 'webcam')
        self.declare_parameter('min_views', 10)
        self.declare_parameter('debug_dir', '')

        # -- same board parameters as capture_node --
        self.declare_parameter('squares_x', 8)
        self.declare_parameter('squares_y', 10)
        self.declare_parameter('square_length_m', 0.015)
        self.declare_parameter('marker_length_m', 0.011)
        self.declare_parameter('dictionary', '4X4_50')
        self.declare_parameter('legacy_pattern', False)
        self.declare_parameter('min_charuco_corners', 6)

        p = self.get_parameter
        self.board, self.detector = build_board(
            p('squares_x').value, p('squares_y').value,
            p('square_length_m').value, p('marker_length_m').value,
            p('dictionary').value, p('legacy_pattern').value)
        self.min_corners = p('min_charuco_corners').value
        self.min_views = p('min_views').value

        output_path = p('output_path').value
        if not output_path:
            self.get_logger().error(
                'Parameter "output_path" is unset -- nowhere to write the result.')
            raise SystemExit(1)
        self.output_path = pathlib.Path(output_path).expanduser()

        debug_dir = p('debug_dir').value
        self.debug_dir = pathlib.Path(debug_dir).expanduser() if debug_dir else None
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Saving annotated views to {self.debug_dir}')

        self.latest_bgr = None
        self.image_size = None      # (width, height)
        self.obj_points = []
        self.img_points = []
        self.frames = []            # kept only to annotate reprojections later

        self.create_subscription(
            Image, p('image_topic').value, self._image_cb, 10)

        self.create_service(Trigger, '~/capture', self._srv_capture)
        self.create_service(Trigger, '~/calibrate', self._srv_calibrate)
        self.create_service(Trigger, '~/reset', self._srv_reset)

        self.get_logger().info(
            'intrinsics_node ready. Services: ~/capture, ~/calibrate, ~/reset')

    def _image_cb(self, msg):
        self.latest_bgr = imgmsg_to_cv2(msg)

    # ------------------------------------------------------------
    def _srv_capture(self, request, response):
        if self.latest_bgr is None:
            response.success = False
            response.message = 'No camera frame received yet.'
            self.get_logger().warn(response.message)
            return response

        bgr = self.latest_bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            self.detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < self.min_corners:
            found = 0 if charuco_ids is None else len(charuco_ids)
            response.success = False
            response.message = (f'Board not detected ({found} corners, '
                                f'need {self.min_corners}).')
            self.get_logger().warn(response.message)
            return response

        obj_pts, img_pts = self.board.matchImagePoints(charuco_corners, charuco_ids)
        self.obj_points.append(obj_pts)
        self.img_points.append(img_pts)
        self.image_size = (gray.shape[1], gray.shape[0])

        n = len(self.obj_points)
        if self.debug_dir is not None:
            self.frames.append(bgr)
            annotated = bgr.copy()
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(
                annotated, charuco_corners, charuco_ids, (0, 0, 255))
            cv2.imwrite(str(self.debug_dir / f'view_{n:03d}.png'), annotated)

        response.success = True
        response.message = f'View {n} captured ({len(charuco_ids)} corners).'
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------
    def _srv_calibrate(self, request, response):
        n = len(self.obj_points)
        if n < 3:
            response.success = False
            response.message = f'Only {n} views -- need at least 3.'
            self.get_logger().error(response.message)
            return response
        if n < self.min_views:
            self.get_logger().warn(
                f'Only {n} views (recommended >= {self.min_views}) -- distortion '
                'terms will be poorly constrained.')

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.image_size, None, None)

        total_error = self._report_per_view(K, dist, rvecs, tvecs)
        self._write_undistorted(K, dist)
        self._write_camera_info(K, dist)

        response.success = True
        response.message = (f'Calibrated over {n} views, total error '
                            f'{total_error:.4f}. Wrote {self.output_path}')
        self.get_logger().info(response.message)
        self.get_logger().info(f'  fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} '
                               f'cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}')
        self.get_logger().info(f'  calibrateCamera rms: {rms:.4f} px')
        if rms > 1.0:
            self.get_logger().warn(
                'calibrateCamera rms > 1.0 px. The per-view table above is '
                'sorted worst first -- a few views far above the rest means '
                'drop those and recapture; every view alike means a systematic '
                'cause (board model, a focus change mid-run, or edge overshoot '
                'from high sharpness).')
        return response

    def _report_per_view(self, K, dist, rvecs, tvecs):
        """Reprojection error per view, worst first; returns the mean over views.

        The error is the one the OpenCV calibration tutorial defines: the L2
        norm of (detected - reprojected) over a view's points, divided by the
        point count, then averaged across views.

        The divisor is N rather than sqrt(N), so this is not an RMS in pixels
        and does not match the rms calibrateCamera returns -- it is smaller by
        roughly sqrt(N), and for the same actual error a view with more corners
        scores lower than one with fewer. Compare views here only when their
        corner counts are close; calibrateCamera's rms, logged separately, is
        the figure to judge in pixels.

        The total alone hides one bad view among the good ones, which is why
        the per-view breakdown is printed: a couple of blurred captures and a
        systematic error affecting every view have completely different fixes.

        projectPoints returns float64 while matchImagePoints gives float32, and
        cv2.norm rejects a mixed pair, so the projection is cast back.
        """
        errors = []
        mean_error = 0
        for i in range(len(self.obj_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i], K, dist)
            imgpoints2 = imgpoints2.astype(self.img_points[i].dtype)
            error = cv2.norm(self.img_points[i], imgpoints2,
                             cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
            errors.append((float(error), i + 1, len(self.img_points[i]),
                           imgpoints2.reshape(-1, 2),
                           self.img_points[i].reshape(-1, 2)))
        total_error = mean_error / len(self.obj_points)

        self.get_logger().info('  per-view reprojection error (worst first):')
        for err, i, n_pts, _, _ in sorted(errors, reverse=True):
            self.get_logger().info(
                f'    view {i:3d}  {n_pts:3d} corners  {err:8.5f}')
        self.get_logger().info(f'  total error: {total_error}')

        if self.debug_dir is not None:
            for err, i, _, proj, img in errors:
                annotated = self.frames[i - 1].copy()
                for (px, py), (ix, iy) in zip(proj, img):
                    cv2.circle(annotated, (int(round(ix)), int(round(iy))), 4,
                               (0, 255, 0), 1)                   # detected
                    cv2.circle(annotated, (int(round(px)), int(round(py))), 2,
                               (0, 0, 255), -1)                  # reprojected
                cv2.putText(annotated, f'view {i}  error {err:.5f}',
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                cv2.imwrite(
                    str(self.debug_dir / f'view_{i:03d}_reproj.png'), annotated)
            self.get_logger().info(
                f'  annotated views written to {self.debug_dir}')

        return total_error

    def _write_undistorted(self, K, dist):
        """Undistort every captured view and re-detect the board on the result.

        Detection is re-run rather than the stored corners reused: if the
        distortion model is right, the board's rows and columns come out
        straight and every marker still decodes. A view where undistortion
        bends the board, or loses markers it had before, is the visible form of
        an over-fitted distortion solve -- which the reprojection error will
        not show, because that is measured in the distorted frame where the
        model was fitted.
        """
        if self.debug_dir is None:
            return
        for i, frame in enumerate(self.frames, start=1):
            undistorted = cv2.undistort(frame, K, dist)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                self.detector.detectBoard(gray)
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    undistorted, marker_corners, marker_ids)
            found = 0
            if charuco_ids is not None:
                found = len(charuco_ids)
                cv2.aruco.drawDetectedCornersCharuco(
                    undistorted, charuco_corners, charuco_ids, (0, 0, 255))
            cv2.putText(undistorted, f'view {i} undistorted  {found} corners',
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(self.debug_dir / f'view_{i:03d}_undistorted.png'),
                        undistorted)
        self.get_logger().info(f'  undistorted views written to {self.debug_dir}')

    def _write_camera_info(self, K, dist):
        """camera_info_manager YAML layout -- this is what usb_cam's
        camera_info_url loads and republishes as CameraInfo."""
        width, height = self.image_size
        d = dist.flatten().tolist()
        text = f"""image_width: {width}
image_height: {height}
camera_name: {self.get_parameter('camera_name').value}
camera_matrix:
  rows: 3
  cols: 3
  data: {[float(x) for x in K.flatten()]}
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {len(d)}
  data: {[float(x) for x in d]}
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: {[float(x) for x in [K[0, 0], 0.0, K[0, 2], 0.0,
                             0.0, K[1, 1], K[1, 2], 0.0,
                             0.0, 0.0, 1.0, 0.0]]}
"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(text, encoding='utf-8')

    # ------------------------------------------------------------
    def _srv_reset(self, request, response):
        self.obj_points.clear()
        self.img_points.clear()
        self.frames.clear()
        response.success = True
        response.message = 'Cleared all stored views.'
        self.get_logger().info(response.message)
        return response


def main():
    rclpy.init()
    node = IntrinsicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
