#!/usr/bin/env python3
"""
process_dataset.py

Offline hand-eye solve from a dataset captured by auto_calibration_node.py.
Re-runs ChArUco detection on the saved images -- so you can freely
re-process with different detection parameters without re-driving the
robot -- then calls the same solve + leave-one-out stability check as
before.

Usage:
  python3 process_dataset.py \
      --dataset_dir /path/to/handeye_dataset \
      --output /path/to/handeye_result.yaml \
      --method DANIILIDIS
"""

import argparse
import os
import yaml
import numpy as np
import cv2
import tf_transformations as tf
from handeye_calibration.utils import build_board, detect_board_pose
from handeye_calibration.solver import (
    solve, leave_one_out_check, METHODS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', default='TSAI',
                         choices=list(METHODS.keys()))
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_dir, 'dataset.yaml')
    with open(dataset_path) as f:
        dataset = yaml.safe_load(f)

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
        image_path = os.path.join(args.dataset_dir, 'images', entry['image_file'])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARNING: could not read {image_path}, skipping.")
            skipped += 1
            continue
        result = detect_board_pose(img, board, detector, camera_matrix,
                                    dist_coeffs, min_corners)
        if result is None:
            print(f"WARNING: no board detected in {entry['image_file']}, skipping.")
            skipped += 1
            continue
        R_target2cam, t_target2cam = result
        R_g2b.append(np.array(entry['R_gripper2base']))
        t_g2b.append(np.array(entry['t_gripper2base']))
        R_t2c.append(R_target2cam)
        t_t2c.append(t_target2cam)

    n = len(R_g2b)
    print(f'Loaded {n} usable samples ({skipped} skipped) from {dataset_path}')
    if n < 8:
        print('WARNING: fewer than 8 samples -- hand-eye solve will likely be '
              'poorly conditioned. Recommend 15-20+.')
    for method in METHODS:
        R, t = solve(R_g2b.copy(), t_g2b.copy(), R_t2c.copy(), t_t2c.copy(), method)
        quat = tf.transforms3d.quaternions.mat2quat(R)  # returns xyzw
        print(f'\n[{method}] gripper -> camera:')
        print(f'  translation (m): {t}')
        print(f'  quaternion (wxyz): {quat}')

        if n >= 6:
            mean_t, std_t = leave_one_out_check(R_g2b, t_g2b, R_t2c, t_t2c, method)
            print('\nLeave-one-out stability check:')
            print(f'  translation std-dev across subsets (m): {std_t}')
            if np.any(std_t > 0.005):
                print('  WARNING: std-dev > 5mm on at least one axis -- check for '
                    'an outlier sample or insufficient rotation diversity.')
            else:
                print('  Looks stable (< 5mm spread on all axes).')

    result = {
        'parent_frame': 'gripper_link', 
        'child_frame': 'camera_color_optical_frame',
        'translation': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
        'rotation_xyzw': {'x': float(quat[1]), 'y': float(quat[2]),
                           'z': float(quat[3]), 'w': float(quat[0])},
        'method': args.method,
        'num_samples': n,
        'num_skipped': skipped,
    }
    with open(args.output, 'w') as f:
        yaml.safe_dump(result, f)
    print(f'\nWrote result to {args.output}')
    print('Try other methods (TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS) on the '
          'same dataset without re-capturing -- compare for agreement.')


if __name__ == '__main__':
    main()
