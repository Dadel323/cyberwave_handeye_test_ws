#!/usr/bin/env python3
"""
solver.py

Reads the raw sample pairs recorded by calibration_collector.py and solves
the eye-in-hand hand-eye problem:

    cv2.calibrateHandEye(R_gripper2base, t_gripper2base,
                          R_target2cam,  t_target2cam,
                          method) -> R_cam2gripper, t_cam2gripper

Writes the resulting gripper->camera static transform to
data/handeye_result.yaml (translation + quaternion, ready to feed into a
static_transform_publisher).

Also runs a leave-one-out stability check: re-solves the calibration with
each sample removed in turn and reports how much the result moves. Large
spread flags either an outlier sample (bad detection, arm not settled) or
a trajectory with insufficient rotation diversity.

Usage:
  python3 solver.py \
      --samples /home/claude/handeye_calibration/data/handeye_samples.yaml \
      --output /home/claude/handeye_calibration/data/handeye_result.yaml \
      --method DANIILIDIS
"""

import argparse
import yaml
import numpy as np
import cv2

# tf_transformations is imported inside main() rather than here: it is only
# needed to write the result YAML, and it lives in the ROS install rather than
# the venv. Importing it at module scope would force every offline analysis
# tool that reuses solve()/leave_one_out_check() to source ROS first.

METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    #'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_samples(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    R_g2b = [np.array(s['R_gripper2base']) for s in raw]
    t_g2b = [np.array(s['t_gripper2base']) for s in raw]
    R_t2c = [np.array(s['R_target2cam']) for s in raw]
    t_t2c = [np.array(s['t_target2cam']) for s in raw]
    return R_g2b, t_g2b, R_t2c, t_t2c


def solve(R_g2b, t_g2b, R_t2c, t_t2c, method_name):
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                 method=METHODS[method_name])
    return R, t.flatten()


def leave_one_out_check(R_g2b, t_g2b, R_t2c, t_t2c, method_name):
    n = len(R_g2b)
    translations = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        try:
            _, t = solve([R_g2b[j] for j in idx], [t_g2b[j] for j in idx],
                         [R_t2c[j] for j in idx], [t_t2c[j] for j in idx],
                         method_name)
            translations.append(t)
        except cv2.error:
            continue
    translations = np.array(translations)
    return translations.mean(axis=0), translations.std(axis=0)


def main():
    import tf_transformations as tf

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', default='DANIILIDIS', choices=list(METHODS.keys()))
    args = parser.parse_args()

    R_g2b, t_g2b, R_t2c, t_t2c = load_samples(args.samples)
    n = len(R_g2b)
    print(f'Loaded {n} samples.')
    if n < 8:
        print('WARNING: fewer than 8 samples -- hand-eye solve will likely be '
              'poorly conditioned. Recommend 15-20+.')

    R, t = solve(R_g2b, t_g2b, R_t2c, t_t2c, args.method)
    
    quat = tf.transforms3d.quaternions.mat2quat(R)  # returns wxyz

    print(f'\n[{args.method}] gripper -> camera:')
    print(f'  translation (m): {t}')
    print(f'  quaternion (wxyz): {quat}')

    if n >= 6:
        mean_t, std_t = leave_one_out_check(R_g2b, t_g2b, R_t2c, t_t2c, args.method)
        print(f'\nLeave-one-out stability check:')
        print(f'  translation std-dev across subsets (m): {std_t}')
        if np.any(std_t > 0.005):
            print('  WARNING: std-dev > 5mm on at least one axis -- check for an '
                  'outlier sample (bad detection / arm not settled) or insufficient '
                  'rotation diversity in the trajectory.')
        else:
            print('  Looks stable (< 5mm spread on all axes).')

    result = {
        'parent_frame': 'gripper_link',   # TODO: confirm matches your URDF
        'child_frame': 'camera_color_optical_frame',  # TODO: confirm matches your URDF
        'translation': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
        'rotation_wxyz': {'w': float(quat[0]), 'x': float(quat[1]),
                           'y': float(quat[2]), 'z': float(quat[3])},
        'method': args.method,
        'num_samples': n,
    }
    with open(args.output, 'w') as f:
        yaml.safe_dump(result, f)
    print(f'\nWrote result to {args.output}')
    print('Compare results across TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS -- if they '
          'agree within a few mm/degrees, that\'s a good sign the calibration is solid.')


if __name__ == '__main__':
    main()
