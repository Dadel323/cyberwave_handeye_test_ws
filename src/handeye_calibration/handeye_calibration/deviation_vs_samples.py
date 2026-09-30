#!/usr/bin/env python3
"""
deviation_vs_samples.py

Offline analysis of a dataset captured by auto_calibration_node.py: plots how
much the hand-eye solvers (TSAI/PARK/HORAUD/DANIILIDIS) disagree with each
other as more samples are used, from --min_samples up to the full dataset.
Each method is solved independently on the same first-n samples, and the
pairwise translation distance between methods is plotted vs. sample count.

Reuses the same board-detection logic as process_dataset.py.

Usage:
  python3 deviation_vs_samples.py \
      --dataset_dir /path/to/handeye_dataset \
      --output deviation.png
"""

import argparse
import itertools
import os

import numpy as np
import cv2
import yaml
import plotly.graph_objects as go

from handeye_calibration.utils import build_board, detect_board_pose
from handeye_calibration.solver import solve, METHODS


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
    parser.add_argument('--min_samples', type=int, default=6,
                         help='hand-eye solve needs at least a few samples to mean anything')
    parser.add_argument('--output', default=None,
                         help='path to save the plot (PNG). Shows interactively if omitted.')
    args = parser.parse_args()

    R_g2b, t_g2b, R_t2c, t_t2c, skipped = load_all_samples(args.dataset_dir)
    n_total = len(R_g2b)
    print(f'Loaded {n_total} usable samples ({skipped} skipped) from {args.dataset_dir}')
    if n_total <= args.min_samples:
        raise SystemExit(f'Need more than {args.min_samples} usable samples, got {n_total}.')

    sample_counts = list(range(args.min_samples, n_total + 1))
    method_names = list(METHODS.keys())
    pairs = list(itertools.combinations(method_names, 2))
    deviation_mm = {pair: [] for pair in pairs}

    for n in sample_counts:
        translations = {}
        for method in method_names:
            try:
                _, t = solve(R_g2b[:n], t_g2b[:n], R_t2c[:n], t_t2c[:n], method)
                translations[method] = t
            except cv2.error:
                continue
        for a, b in pairs:
            if a in translations and b in translations:
                dist_mm = np.linalg.norm(translations[a] - translations[b]) * 1000.0
                deviation_mm[(a, b)].append(dist_mm)
            else:
                deviation_mm[(a, b)].append(np.nan)

    fig = go.Figure()
    for (a, b) in pairs:
        fig.add_trace(go.Scatter(x=sample_counts, y=deviation_mm[(a, b)],
                                  mode='lines+markers', name=f'{a} vs {b}'))
    fig.update_layout(
        xaxis_title='Number of samples used',
        yaxis_title='Translation deviation between methods (mm)',
        title=f'Hand-eye method agreement vs. sample count ({n_total} total samples)')

    if args.output:
        fig.write_image(args.output)
        print(f'Saved plot to {args.output}')
    else:
        fig.show()


if __name__ == '__main__':
    main()
