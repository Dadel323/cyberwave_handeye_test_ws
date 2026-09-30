#!/usr/bin/env python3
"""
visualize_board_poses.py

Re-detects the ChArUco board in every saved calibration image (same
detection code path as process_dataset.py) and draws the board's
reprojected coordinate axes + corner IDs onto each image. This lets you
visually confirm the detected pose is sane -- a flipped/mirrored
detection (a known ChArUco failure mode at certain viewing angles) will
show the axes pointing the wrong way or twisted relative to the visible
board, which is very obvious to the eye but invisible in the raw
R_target2cam numbers alone.

Also builds a single contact-sheet image (grid of thumbnails) so you can
scan all samples at a glance instead of opening each file individually.

Usage:
  python3 visualize_board_poses.py \
      --dataset_dir src/handeye_calibration/data/ \
      --output_dir /tmp/board_pose_check

Then view /tmp/board_pose_check/contact_sheet.png (or the individual
per-sample images in the same directory).
"""

import argparse
import os
import math

import numpy as np
import cv2
import yaml

from handeye_calibration.utils import build_board, detect_board_pose


def annotate_image(img_gray, board, detector, camera_matrix, dist_coeffs,
                    min_corners, axis_length_m):
    """Returns a BGR image with detected corners + reprojected axes drawn
    on it, plus the (R, t) pose or None if detection failed."""
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(img_gray)
    if charuco_ids is None or len(charuco_ids) < min_corners:
        cv2.putText(img_color, 'NO DETECTION', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return img_color, None

    cv2.aruco.drawDetectedCornersCharuco(img_color, charuco_corners, charuco_ids)

    result = detect_board_pose(img_gray, board, detector, camera_matrix,
                                dist_coeffs, min_corners)
    if result is None:
        cv2.putText(img_color, 'POSE SOLVE FAILED', (20, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return img_color, None

    R, t = result
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    cv2.drawFrameAxes(img_color, camera_matrix, dist_coeffs, rvec, tvec,
                       axis_length_m, thickness=4)

    # Annotate distance from camera + a rough "facing" sanity number:
    # z-axis of the board expressed in camera frame -- if this dot product
    # with the camera's own +z (viewing direction) is strongly negative,
    # the board is being viewed from behind/mirrored, which is suspicious.
    board_z_in_cam = R[:, 2]
    facing_dot = float(board_z_in_cam[2])  # cam looks down +z; board normal's z-component
    dist_m = float(np.linalg.norm(t))

    label = f'dist={dist_m:.3f}m  facing_dot={facing_dot:+.2f}'
    color = (0, 255, 0) if facing_dot < 0 else (0, 165, 255)  # orange = suspicious
    cv2.putText(img_color, label, (20, img_color.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return img_color, (R, t, facing_dot)


def build_contact_sheet(images, labels, thumb_size=(320, 240), cols=6):
    n = len(images)
    if n == 0:
        return None
    rows = math.ceil(n / cols)
    tw, th = thumb_size
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    for idx, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(img, (tw, th))
        cv2.putText(thumb, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb

    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--axis_length_m', type=float, default=None,
                         help='Defaults to 3x the board square length if not set.')
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

    axis_length_m = args.axis_length_m or (3 * board_cfg['square_length_m'])

    os.makedirs(args.output_dir, exist_ok=True)

    thumbs, labels = [], []
    suspicious = []

    for entry in dataset['entries']:
        image_file = entry['image_file']
        image_path = os.path.join(args.dataset_dir, 'images', image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'WARNING: could not read {image_path}, skipping.')
            continue

        annotated, pose_info = annotate_image(
            img, board, detector, camera_matrix, dist_coeffs,
            min_corners, axis_length_m)

        out_path = os.path.join(args.output_dir, f'annotated_{image_file}')
        cv2.imwrite(out_path, annotated)

        if pose_info is None:
            label = f'{image_file}: NO DETECTION'
        else:
            _, _, facing_dot = pose_info
            label = f'{image_file}'
            if facing_dot >= 0:
                suspicious.append(image_file)
                label += ' (CHECK ME)'

        thumbs.append(annotated)
        labels.append(label)
        print(f'{image_file}: {"NO DETECTION" if pose_info is None else "ok"}')

    sheet = build_contact_sheet(thumbs, labels)
    if sheet is not None:
        sheet_path = os.path.join(args.output_dir, 'contact_sheet.png')
        cv2.imwrite(sheet_path, sheet)
        print(f'\nWrote contact sheet: {sheet_path}')

    print(f'Annotated {len(thumbs)} images -> {args.output_dir}')
    if suspicious:
        print(f'\n{len(suspicious)} sample(s) flagged as possibly flipped/mirrored '
              f'(facing_dot >= 0 -- board normal pointing away from camera in an '
              f'unexpected direction). Look at these first:')
        for f in suspicious:
            print(f'  - {f}')
    else:
        print('\nNo samples flagged by the facing_dot heuristic -- still worth a '
              'quick visual scan of the contact sheet, since this check only '
              'catches gross flips, not subtle corner-order errors.')


if __name__ == '__main__':
    main()