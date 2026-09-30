import os
import yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
import tf_transformations as tf
from handeye_calibration_interfaces.srv import CalibrateDataset
from handeye_calibration.utils import build_board, detect_board_pose
from handeye_calibration.solver import (
    solve, leave_one_out_check, METHODS)


class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')
        # The solved transform is gripper -> camera; these name those two frames
        # in the result file. The camera frame changes with the camera, so it is
        # a parameter rather than the hard-coded Orbbec frame it used to be.
        self.declare_parameter('parent_frame', 'openarm_right_hand')
        self.declare_parameter('child_frame', 'webcam_optical_frame')

        self._service = self.create_service(
            CalibrateDataset, 'calibrate_dataset', self.calibrate_callback)
        self.get_logger().info('calibration_node ready, service: calibrate_dataset')

    def calibrate_callback(self, request, response):
        dataset_dir = request.dataset_dir
        method = request.method if request.method in METHODS else 'DANIILIDIS'

        dataset_path = os.path.join(dataset_dir, 'dataset.yaml')
        try:
            with open(dataset_path) as f:
                dataset = yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().error(f'Dataset not found: {dataset_path}')
            response.success = False
            return response

        camera_matrix = np.array(dataset['camera_matrix'])
        dist_coeffs = np.array(dataset['dist_coeffs'])
        board_cfg = dataset['board']
        board, detector = build_board(
            board_cfg['squares_x'], board_cfg['squares_y'],
            board_cfg['square_length_m'], board_cfg['marker_length_m'],
            board_cfg['dictionary'], board_cfg['legacy_pattern'])
        min_corners = board_cfg['min_charuco_corners']

        R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
        skipped = 0
        for entry in dataset['entries']:
            image_path = os.path.join(dataset_dir, 'images', entry['image_file'])
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue
            result = detect_board_pose(img, board, detector, camera_matrix,
                                        dist_coeffs, min_corners)
            if result is None:
                skipped += 1
                continue
            R_target2cam, t_target2cam = result
            R_g2b.append(np.array(entry['R_gripper2base']))
            t_g2b.append(np.array(entry['t_gripper2base']))
            R_t2c.append(R_target2cam)
            t_t2c.append(t_target2cam)

        n = len(R_g2b)
        self.get_logger().info(f'Loaded {n} usable samples ({skipped} skipped)')
        if n < 8:
            self.get_logger().warn(
                'Fewer than 8 samples -- solve will likely be poorly conditioned.')
        if n < 3:
            response.success = False
            return response

        R, t = solve(R_g2b, t_g2b, R_t2c, t_t2c, method)
        # transforms3d's mat2quat returns (w, x, y, z); this file writes
        # rotation_xyzw. Reading its output positionally as x,y,z,w shifted
        # every component by one, which turned the solved -179.97 deg yaw into
        # +11.57 deg -- the camera frame appeared rotated ~180 deg in RViz.
        qw, qx, qy, qz = tf.transforms3d.quaternions.mat2quat(R) 
        quat = (qx, qy, qz, qw)

        stddev = np.zeros(3)
        if n >= 6:
            _, stddev = leave_one_out_check(R_g2b, t_g2b, R_t2c, t_t2c, method)

        result_yaml_path = os.path.join(dataset_dir, 'handeye_result.yaml')
        result_data = {
            'parent_frame': self.get_parameter('parent_frame').value,
            'child_frame': self.get_parameter('child_frame').value,
            'translation': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
            'rotation_xyzw': {'x': float(quat[0]), 'y': float(quat[1]),
                               'z': float(quat[2]), 'w': float(quat[3])},
            'method': method,
            'num_samples': n,
            'num_skipped': skipped,
        }
        with open(result_yaml_path, 'w') as f:
            yaml.safe_dump(result_data, f)

        response.success = True
        response.result_yaml_path = result_yaml_path
        response.translation = [float(x) for x in t]
        response.rotation_xyzw = [float(x) for x in quat]
        response.stability_stddev = [float(x) for x in stddev]
        return response


def main():
    rclpy.init()
    node = CalibrationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()