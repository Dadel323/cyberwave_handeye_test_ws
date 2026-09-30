#!/usr/bin/env python3
"""
visualize_scene.py

Draws the whole hand-eye calibration as one 3D scene in base coordinates:
the robot base, the gripper pose at every sample, the camera riding on the
gripper via the solved transform, and the board pose each sample implies.

The point of the picture is the board cluster. Hand-eye rests on one
invariant -- the board does not move, so every sample must agree about
where it is:

    board_in_base_i = A_i @ X @ B_i

        A_i  base -> gripper   (recorded at capture time)
        X    gripper -> camera (what calibrateHandEye solves for)
        B_i  camera -> board   (re-detected from the sample image)

If X is right those poses land on top of each other. If X is wrong they
spray out, and the spread is a direct measure of the error in metres --
unlike the leave-one-out check, which only says the answer is unstable,
not how far off it is or which capture is dragging it.

What this shows that the numbers do not:

  * a board cluster that is tight but the wrong shape -- the arm never
    rotated enough, so X is underdetermined in one direction
  * one capture flung far from the rest -- a bad detection to drop
  * frustums that all point from nearly the same place, which is the
    classic under-conditioned trajectory that still reports a small
    residual

Usage:
  python3 -m handeye_calibration.visualize_scene \
      --dataset_dir src/handeye_calibration/data_so101 \
      --method DANIILIDIS \
      --output scene.png
"""

import argparse

import numpy as np
import matplotlib

from handeye_calibration.utils import load_dataset
from handeye_calibration.solver import solve, leave_one_out_check, METHODS


# --- geometry --------------------------------------------------------------


def homogeneous(R, t):
    """3x3 rotation + 3-vector translation -> 4x4 rigid transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def rotation_angle_deg(R):
    """Geodesic angle of a rotation, in degrees. Clipped because a matrix
    built from float arithmetic can put the trace a hair outside acos'
    domain."""
    cos_angle = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def mean_rotation(rotations):
    """Average of rotations that already sit close together.

    Averaged as quaternions with the signs aligned first: q and -q are the
    same rotation, so a raw mean of mixed signs cancels toward zero and can
    produce a 'mean' pointing nowhere near any input.
    """
    quats = []
    reference = None
    for R in rotations:
        q = _mat_to_quat_wxyz(R)
        if reference is None:
            reference = q
        elif np.dot(q, reference) < 0.0:
            q = -q
        quats.append(q)
    mean = np.mean(quats, axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        return rotations[0]
    return _quat_wxyz_to_mat(mean / norm)


def _mat_to_quat_wxyz(R):
    """Rotation matrix -> (w, x, y, z).

    Branches on the largest diagonal term rather than always using the
    trace, so the divisor stays away from zero near a half turn -- which is
    exactly where a wrist-mounted camera lives, the optical-frame flip
    being a half turn about X.

    (Written out rather than imported from tf_transformations so this tool
    runs without a sourced ROS environment. Note that transforms3d's
    mat2quat also returns wxyz, despite solver.py:91 commenting otherwise.)
    """
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# --- drawing ---------------------------------------------------------------

AXIS_COLORS = ('#d62728', '#2ca02c', '#1f77b4')  # x red, y green, z blue


def draw_frame(ax, T, length, alpha=1.0, linewidth=1.6):
    """Coordinate triad for a 4x4 pose."""
    origin = T[:3, 3]
    for axis in range(3):
        tip = origin + T[:3, axis] * length
        ax.plot(*zip(origin, tip), color=AXIS_COLORS[axis],
                alpha=alpha, linewidth=linewidth)


def draw_frustum(ax, T, camera_matrix, image_size, depth, color, alpha=0.55):
    """Camera viewing cone, sized by the real intrinsics.

    The image corners are back-projected through K rather than drawn as an
    arbitrary wedge, so the cone's opening angle is the camera's actual
    field of view. A frustum that visibly misses the board is the fastest
    way to spot a wrong X.
    """
    width, height = image_size
    corners_px = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                          dtype=float)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # OpenCV optical frame: +z along the view axis, so depth scales directly.
    rays = np.stack([
        (corners_px[:, 0] - cx) / fx * depth,
        (corners_px[:, 1] - cy) / fy * depth,
        np.full(4, depth),
    ], axis=1)

    apex = T[:3, 3]
    corners = (T[:3, :3] @ rays.T).T + apex
    for corner in corners:
        ax.plot(*zip(apex, corner), color=color, alpha=alpha, linewidth=0.9)
    loop = np.vstack([corners, corners[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color,
            alpha=alpha, linewidth=0.9)


def draw_board(ax, T, board_cfg, color, alpha=0.30):
    """The board as a filled quad at its true physical size.

    The board's origin is its first inner corner, not its centre, and its
    plane spans +x/+y in its own frame -- so the quad is drawn from the
    origin outward, matching how solvePnP reports the pose.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    width = board_cfg['squares_x'] * board_cfg['square_length_m']
    height = board_cfg['squares_y'] * board_cfg['square_length_m']
    local = np.array([
        [0.0, 0.0, 0.0],
        [width, 0.0, 0.0],
        [width, height, 0.0],
        [0.0, height, 0.0],
    ])
    corners = (T[:3, :3] @ local.T).T + T[:3, 3]
    ax.add_collection3d(Poly3DCollection(
        [corners], facecolor=color, edgecolor=color,
        alpha=alpha, linewidths=0.8))


def set_equal_aspect(ax, points):
    """Force a cube aspect around the data.

    Without this matplotlib stretches each axis independently and a tight
    board cluster can be rendered looking like a wide spray, which would
    invert the tool's entire message.
    """
    points = np.asarray(points)
    centre = (points.max(axis=0) + points.min(axis=0)) / 2.0
    radius = float((points.max(axis=0) - points.min(axis=0)).max()) / 2.0
    radius = max(radius, 0.05)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:  # matplotlib < 3.3
        pass


# --- analysis --------------------------------------------------------------


def implied_board_poses(samples, X):
    """board_in_base for every sample: A @ X @ B."""
    return [homogeneous(s['R_g2b'], s['t_g2b']) @ X
            @ homogeneous(s['R_t2c'], s['t_t2c']) for s in samples]


def deviations(poses):
    """Each pose's distance from the consensus, in metres and degrees.

    The consensus is the mean board pose. This is the residual that matters:
    it is expressed in the units of the thing being measured, so 30 mm here
    means the calibration is wrong by about 30 mm.
    """
    reference_t = np.mean([T[:3, 3] for T in poses], axis=0)
    reference_R = mean_rotation([T[:3, :3] for T in poses])
    reference_R_inv = reference_R.T
    return [
        {
            'translation_m': float(np.linalg.norm(T[:3, 3] - reference_t)),
            'rotation_deg': rotation_angle_deg(reference_R_inv @ T[:3, :3]),
        }
        for T in poses
    ], homogeneous(reference_R, reference_t)


# --- main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--method', default='DANIILIDIS',
                        choices=list(METHODS.keys()))
    parser.add_argument('--output', default=None,
                        help='path to save the plot (PNG). Shows interactively if omitted.')
    parser.add_argument('--frustum_depth', type=float, default=0.06,
                        help='how far to draw the camera cones, metres (cosmetic)')
    args = parser.parse_args()

    if args.output:
        # cv2 (pulled in by utils, needed for board detection) points Qt at
        # its own bundled plugins, which breaks matplotlib's default backend
        # (missing xcb platform plugin). Agg is a headless, file-only
        # backend, so it sidesteps that conflict entirely.
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Registers the '3d' projection. Recent matplotlib no longer imports it as
    # a side effect of pyplot, so add_subplot(projection='3d') fails without it.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    samples, meta = load_dataset(args.dataset_dir)
    n = len(samples)
    print(f'Loaded {n} usable samples of {meta["total_entries"]} '
          f'({len(meta["skipped"])} skipped) from {args.dataset_dir}')
    if meta['skipped']:
        for index, image_file, reason in meta['skipped']:
            print(f'  skipped sample {index:3d} ({image_file}): {reason}')
    if n < 3:
        raise SystemExit(f'Need at least 3 usable samples to draw a scene, got {n}.')

    R_g2b = [s['R_g2b'] for s in samples]
    t_g2b = [s['t_g2b'] for s in samples]
    R_t2c = [s['R_t2c'] for s in samples]
    t_t2c = [s['t_t2c'] for s in samples]

    R_x, t_x = solve(R_g2b, t_g2b, R_t2c, t_t2c, args.method)
    X = homogeneous(R_x, t_x)
    print(f'\n[{args.method}] gripper -> camera:')
    print(f'  translation (m): [{t_x[0]:+.5f} {t_x[1]:+.5f} {t_x[2]:+.5f}]')
    quat = _mat_to_quat_wxyz(R_x)
    print(f'  quaternion (wxyz): [{quat[0]:+.5f} {quat[1]:+.5f} '
          f'{quat[2]:+.5f} {quat[3]:+.5f}]')

    poses = implied_board_poses(samples, X)
    devs, reference = deviations(poses)

    print('\nBoard pose implied by each sample (deviation from consensus):')
    print(f'{"sample":>8}  {"image":<18} {"translation":>12}  {"rotation":>10}')
    order = sorted(range(n), key=lambda i: -devs[i]['translation_m'])
    for rank, i in enumerate(order):
        mark = '  <-- worst' if rank == 0 else ''
        print(f'{samples[i]["index"]:>8}  {samples[i]["image_file"]:<18} '
              f'{devs[i]["translation_m"] * 1000:>9.1f} mm  '
              f'{devs[i]["rotation_deg"]:>7.2f} deg{mark}')

    spread_mm = np.array([d['translation_m'] for d in devs]) * 1000
    print(f'\nboard-pose spread: mean {spread_mm.mean():.1f} mm, '
          f'worst {spread_mm.max():.1f} mm')
    print('  (this is the calibration error in the units you care about; a good '
          'result is a\n   few mm. It cannot detect an error that is consistent '
          'across every sample,\n   such as a wrong square size, which rescales '
          'the whole scene self-consistently.)')

    if n >= 6:
        _, std_t = leave_one_out_check(R_g2b, t_g2b, R_t2c, t_t2c, args.method)
        print(f'leave-one-out translation stddev (mm): '
              f'[{std_t[0] * 1000:.2f} {std_t[1] * 1000:.2f} {std_t[2] * 1000:.2f}]')

    # --- draw ---
    fig = plt.figure(figsize=(13, 6.5))
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=float(spread_mm.min()), vmax=float(spread_mm.max()))
    image_size = (meta['camera_matrix'][0, 2] * 2, meta['camera_matrix'][1, 2] * 2)
    gripper_axis = 0.03

    all_points = [np.zeros(3)]
    for T in poses:
        all_points.append(T[:3, 3])
    for s in samples:
        all_points.append(np.asarray(s['t_g2b']).reshape(3))

    # Left: the whole scene. Right: the board cluster on its own, which is
    # where the answer actually lives and is otherwise too small to read.
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    draw_frame(ax, np.eye(4), 0.05, linewidth=2.6)
    ax.text(0, 0, 0, '  base_link', fontsize=8)

    for i, s in enumerate(samples):
        A = homogeneous(s['R_g2b'], s['t_g2b'])
        camera = A @ X
        color = cmap(norm(spread_mm[i]))
        draw_frame(ax, A, gripper_axis, alpha=0.85)
        draw_frustum(ax, camera, meta['camera_matrix'], image_size,
                     args.frustum_depth, color)
        draw_board(ax, poses[i], meta['board'], color)

    set_equal_aspect(ax, all_points)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(f'Scene in base_link  --  {n} samples, {args.method}\n'
                 'triads = gripper, cones = camera, quads = implied board',
                 fontsize=10, pad=14)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for i in range(n):
        color = cmap(norm(spread_mm[i]))
        draw_board(ax2, poses[i], meta['board'], color, alpha=0.22)
        ax2.scatter(*poses[i][:3, 3], color=color, s=36, depthshade=False)
        ax2.text(*poses[i][:3, 3], f'  {samples[i]["index"]}', fontsize=7)
    draw_frame(ax2, reference, 0.02, linewidth=2.2)

    set_equal_aspect(ax2, [T[:3, 3] for T in poses])
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('z (m)')
    ax2.set_title(f'Board agreement (tight = good)\n'
                  f'mean {spread_mm.mean():.1f} mm, worst {spread_mm.max():.1f} mm',
                  fontsize=10)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax2, shrink=0.65, label='deviation (mm)')

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f'\nSaved plot to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
