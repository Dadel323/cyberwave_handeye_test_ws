#!/usr/bin/env python3
"""
accuracy_vs_samples.py

Offline analysis of a dataset captured by auto_calibration_node.py: plots how
the leave-one-out translation stddev (the same stability check
process_dataset.py/calibration_node.py already report) changes as more
samples are used, from --min_samples up to the full dataset.

Reuses the same board-detection + leave-one-out logic as process_dataset.py.

Usage:
  python3 accuracy_vs_samples.py \
      --dataset_dir /path/to/handeye_dataset \
      --method DANIILIDIS \
      --output convergence.png
"""

import argparse
import os

import numpy as np
import cv2
import yaml
import matplotlib

from handeye_calibration.utils import build_board, detect_board_pose
from handeye_calibration.solver import leave_one_out_check, METHODS


def load_all_samples(dataset_dir):
    dataset_path = os.path.join(dataset_dir, 'dataset.yaml')
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

    return R_g2b, t_g2b, R_t2c, t_t2c, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--method', default='DANIILIDIS', choices=list(METHODS.keys()))
    parser.add_argument('--min_samples', type=int, default=6,
                         help='leave-one-out needs at least a few samples to mean anything')
    parser.add_argument('--output', default=None,
                         help='path to save the plot (PNG). Shows interactively if omitted.')
    args = parser.parse_args()

    if args.output:
        # cv2 (imported above, needed for board detection) points Qt at its
        # own bundled plugins, which breaks matplotlib's default GUI backend
        # (missing xcb platform plugin). Agg is a headless, file-only
        # backend, so it sidesteps that conflict entirely.
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    R_g2b, t_g2b, R_t2c, t_t2c, skipped = load_all_samples(args.dataset_dir)
    n_total = len(R_g2b)
    print(f'Loaded {n_total} usable samples ({skipped} skipped) from {args.dataset_dir}')
    if n_total <= args.min_samples:
        raise SystemExit(f'Need more than {args.min_samples} usable samples, got {n_total}.')

    sample_counts = list(range(args.min_samples, n_total + 1))
    stddev_norm_mm = []

    for n in sample_counts:
        _, std_t = leave_one_out_check(
            R_g2b[:n], t_g2b[:n], R_t2c[:n], t_t2c[:n], args.method)
        stddev_norm_mm.append(np.linalg.norm(std_t) * 1000)
        print(f'N={n:3d}: leave-one-out translation stddev {stddev_norm_mm[-1]:.3f} mm')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sample_counts, stddev_norm_mm, marker='o')
    ax.set_xlabel('Number of samples used')
    ax.set_ylabel('Leave-one-out translation\nstddev (mm)')
    ax.set_title(f'Hand-eye calibration stability vs. sample count ({args.method}, {n_total} total samples)')
    ax.grid(True)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f'Saved plot to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
