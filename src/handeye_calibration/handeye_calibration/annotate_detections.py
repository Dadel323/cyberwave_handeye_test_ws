#!/usr/bin/env python3
"""
annotate_detections.py

Draw the ChArUco detection onto every image of a capture dataset so you can
see, per frame, which markers were recognised and where the board coordinate
frame ended up.

Each output image gets:
  * the detected ArUco markers, outlined with their ids
  * the interpolated ChArUco chessboard corners, with their ids
  * the board origin axes (X red, Y green, Z blue) drawn at the pose that
    detect_board_pose() solves -- i.e. exactly the pose the calibration uses
  * the board outline (orange), which stays visible on close-up frames where
    the origin projects outside the image
  * a caption with the corner count and the pose that was solved

Frames where the board was rejected are still written out (with whatever
partial detection there was) and flagged, since those are the ones worth
looking at.

Run it with the same interpreter as the other offline dataset scripts (the
venv, not `ros2 run` -- that uses /usr/bin/python3, whose OpenCV 4.6 predates
the CharucoParameters API that utils.build_board needs):

  python3 -m handeye_calibration.annotate_detections \
      --dataset_dir /path/to/data_so101 \
      --output_dir /path/to/annotated

  # look at one frame only, on screen
  python3 -m handeye_calibration.annotate_detections \
      --dataset_dir ... --index 7 --show
"""

import argparse
import os

import cv2
import numpy as np
import yaml

from cv2 import aruco

from handeye_calibration.utils import build_board, detect_board_pose


def board_extent(board):
    """Board width/height in metres, from its own chessboard corner grid.

    Derived rather than recomputed from squares_x * square_length so it stays
    correct for whichever pattern (legacy or not) the board was built with.
    """
    corners = board.getChessboardCorners()
    return corners[:, 0].max(), corners[:, 1].max()


def draw_board_outline(image, board, rvec, tvec, camera_matrix, dist_coeffs):
    """Outline the board plane, so the solved pose stays visible on close-up
    frames where the origin itself projects outside the image."""
    w, h = board_extent(board)
    plane = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]],
                     dtype=np.float64)
    pts, _ = cv2.projectPoints(plane, rvec, tvec, camera_matrix, dist_coeffs)
    cv2.polylines(image, [pts.reshape(-1, 2).astype(np.int32)], True,
                  (0, 165, 255), 2, cv2.LINE_AA)


def draw_origin_marker(image, rvec, tvec, camera_matrix, dist_coeffs):
    """Mark the board origin. drawFrameAxes labels nothing, and with the board
    flat on a table X/Y are easy to confuse.

    When the origin projects outside the image -- common on close-up frames --
    draw an arrow from the nearest edge pointing at it instead, so the frame
    location is still readable rather than silently missing.
    """
    h, w = image.shape[:2]
    origin, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, camera_matrix,
                                  dist_coeffs)
    ox, oy = origin.reshape(2)
    if not np.all(np.isfinite([ox, oy])):
        return
    ox, oy = int(round(ox)), int(round(oy))

    if 0 <= ox < w and 0 <= oy < h:
        cv2.circle(image, (ox, oy), 5, (255, 255, 255), -1)
        cv2.putText(image, 'origin', (ox + 8, oy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return

    # Off-image: clamp into a margin and point back toward the true position.
    margin = 30
    cx, cy = np.clip(ox, margin, w - margin), np.clip(oy, margin, h - margin)
    direction = np.array([ox - cx, oy - cy], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        tip = np.array([cx, cy]) + direction / norm * 22
        cv2.arrowedLine(image, (cx, cy), tuple(tip.astype(int)),
                        (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(image, 'origin off-image', (cx - 60, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def annotate(image_bgr, board, detector, camera_matrix, dist_coeffs,
             min_corners, axis_length_m):
    """Draw the detection onto a copy of image_bgr.

    Returns (annotated_image, info) where info describes what was found:
    n_markers, n_corners, pose (R, t) or None, and ok -- whether this frame
    would be accepted by the calibration.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    out = image_bgr.copy()

    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        detector.detectBoard(gray)

    n_markers = 0 if marker_ids is None else len(marker_ids)
    n_corners = 0 if charuco_ids is None else len(charuco_ids)

    # Markers first, so the chessboard corners drawn next stay on top.
    if n_markers:
        aruco.drawDetectedMarkers(out, marker_corners, marker_ids,
                                  borderColor=(0, 255, 255))
    if n_corners:
        aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids,
                                         cornerColor=(255, 0, 255))

    # Re-solve through the shared helper rather than reimplementing the PnP
    # here, so the axes show the pose the calibration actually consumes.
    pose = detect_board_pose(gray, board, detector, camera_matrix, dist_coeffs,
                             min_corners)
    if pose is not None:
        R, t = pose
        rvec, _ = cv2.Rodrigues(R)
        draw_board_outline(out, board, rvec, t, camera_matrix, dist_coeffs)
        cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvec, t,
                          axis_length_m)
        draw_origin_marker(out, rvec, t, camera_matrix, dist_coeffs)

    info = {
        'n_markers': n_markers,
        'n_corners': n_corners,
        'pose': pose,
        'ok': pose is not None,
    }
    return out, info


def draw_caption(image, lines, ok):
    """Stamp status text in the top-left corner over a dark strip."""
    pad, line_h = 6, 20
    height = pad * 2 + line_h * len(lines)
    width = min(image.shape[1], 520)
    strip = image[0:height, 0:width]
    cv2.addWeighted(strip, 0.35, np.zeros_like(strip), 0.65, 0, strip)

    color = (0, 255, 0) if ok else (0, 0, 255)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (pad, pad + line_h * (i + 1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description='Annotate dataset images with ChArUco detections.')
    parser.add_argument('--dataset_dir', required=True,
                        help='Dataset directory containing dataset.yaml and images/.')
    parser.add_argument('--output_dir', default=None,
                        help='Where to write annotated images '
                             '(default: <dataset_dir>/annotated).')
    parser.add_argument('--index', type=int, default=None,
                        help='Annotate only this entry index from dataset.yaml.')
    parser.add_argument('--axis_length_m', type=float, default=None,
                        help='Length of the drawn frame axes in metres '
                             '(default: 2 * square_length_m).')
    parser.add_argument('--show', action='store_true',
                        help='Display each annotated image; any key advances, '
                             'q or ESC quits.')
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_dir, 'dataset.yaml')
    with open(dataset_path) as f:
        dataset = yaml.safe_load(f)

    if 'board' not in dataset:
        parser.error(f'{dataset_path} has no "board" section -- this dataset '
                     'was not captured for ChArUco calibration.')

    camera_matrix = np.array(dataset['camera_matrix'])
    dist_coeffs = np.array(dataset['dist_coeffs'])
    board_cfg = dataset['board']
    board, detector = build_board(
        board_cfg['squares_x'], board_cfg['squares_y'],
        board_cfg['square_length_m'], board_cfg['marker_length_m'],
        board_cfg['dictionary'], board_cfg['legacy_pattern'])
    min_corners = board_cfg['min_charuco_corners']
    axis_length_m = args.axis_length_m or 2.0 * board_cfg['square_length_m']

    output_dir = args.output_dir or os.path.join(args.dataset_dir, 'annotated')
    os.makedirs(output_dir, exist_ok=True)

    entries = list(enumerate(dataset['entries']))
    if args.index is not None:
        if not 0 <= args.index < len(entries):
            parser.error(f'--index {args.index} out of range '
                         f'(dataset has {len(entries)} entries).')
        entries = [entries[args.index]]

    n_ok, n_failed = 0, 0
    for index, entry in entries:
        image_file = entry['image_file']
        image_path = os.path.join(args.dataset_dir, 'images', image_file)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'[{index:3d}] {image_file}: UNREADABLE, skipping.')
            n_failed += 1
            continue

        out, info = annotate(img, board, detector, camera_matrix, dist_coeffs,
                             min_corners, axis_length_m)

        lines = [f'#{index}  {image_file}',
                 f'markers: {info["n_markers"]}   '
                 f'charuco corners: {info["n_corners"]} (min {min_corners})']
        if info['ok']:
            R, t = info['pose']
            rvec, _ = cv2.Rodrigues(R)
            lines.append(f't (m): [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]')
            lines.append(f'rvec:  [{rvec[0, 0]:+.3f} {rvec[1, 0]:+.3f} '
                         f'{rvec[2, 0]:+.3f}]')
            n_ok += 1
        else:
            lines.append('POSE NOT SOLVED -- frame dropped by calibration')
            n_failed += 1
        draw_caption(out, lines, info['ok'])

        out_path = os.path.join(output_dir, image_file)
        cv2.imwrite(out_path, out)
        print(f'[{index:3d}] {image_file}: markers={info["n_markers"]:3d} '
              f'corners={info["n_corners"]:3d} '
              f'{"ok" if info["ok"] else "NO POSE"} -> {out_path}')

        if args.show:
            cv2.imshow('charuco detection', out)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):
                break
    if args.show:
        cv2.destroyAllWindows()

    print(f'\n{n_ok} frames with a solved pose, {n_failed} without.')
    print(f'Annotated images in {output_dir}')


if __name__ == '__main__':
    main()
