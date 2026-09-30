import numpy as np
import cv2
import pathlib


def write_ply(path, points, colors):
    """Plain ASCII .ply writer -- no external point-cloud library
    needed. (Open3D was tried but isn't installable on this Jetson
    setup -- no matching wheel found via pip, and no apt package
    either.) `colors` is already RGB order, 0-255 range uint8."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f'{x} {y} {z} {int(r)} {int(g)} {int(b)}\n')


def reconstruct_point_cloud(dataset_dir, manifest, n_features=4000, ratio_test=0.75,
                             min_matches=8, max_dist=2.0, out_dir=None):
    """
    Sparse point-cloud reconstruction via ORB feature matching +
    triangulation across all image pairs, using each image's KNOWN
    camera pose. Shared by both reconstruct_3d_node (ROS) and the
    standalone script -- no ROS dependency in this module.

    dataset_dir: where images/ and manifest live (input).
    out_dir: where to write reconstruction.ply (output). Defaults to
        dataset_dir if not given, so dataset_dir doubles as both input
        and output location unless you want them separated.
    """
    dataset_dir = pathlib.Path(dataset_dir).expanduser()
    images_dir = dataset_dir / 'images'

    if out_dir is None:
        out_dir = dataset_dir
    else:
        out_dir = pathlib.Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    K = np.array(manifest['camera_matrix'])
    dist = np.array(manifest['dist_coeffs'])
    entries = manifest['entries']

    if len(entries) < 2:
        print('Need at least 2 samples to reconstruct anything.')
        return None

    orb = cv2.ORB_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Precompute per-image data: features + known projection matrix.
    # P maps a homogeneous point in base_link frame -> homogeneous
    # pixel coords for THIS image (all images share the same
    # base_link "world" frame, which is what makes triangulating
    # across arbitrary pairs valid).
    images = []
    for entry in entries:
        img = cv2.imread(str(images_dir / entry['image_file']))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        R_c2b = np.array(entry['R_camera2base'])
        t_c2b = np.array(entry['t_camera2base'])
        # invert camera->base (what we stored) to get base->camera
        # (what a projection matrix needs: world points -> this
        # camera's pixels)
       
        P = K @ np.hstack([R_c2b, t_c2b.reshape(3, 1)])

        images.append({'img': img, 'kp': kp, 'des': des, 'P': P})

    all_points = []
    all_colors = []
    n = len(images)
    n_pairs_used = 0

    for i in range(n):
        for j in range(i + 1, n):
            a, b = images[i], images[j]
            if a['des'] is None or b['des'] is None:
                continue

            matches = bf.knnMatch(a['des'], b['des'], k=2)
            good = [m for m, m2 in matches
                    if len(matches) and m.distance < ratio_test * m2.distance]
            if len(good) < min_matches:
                continue

            pts_a = np.array([a['kp'][m.queryIdx].pt for m in good], dtype=np.float64)
            pts_b = np.array([b['kp'][m.trainIdx].pt for m in good], dtype=np.float64)

            pts_a_ud = cv2.undistortPoints(
                pts_a.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
            pts_b_ud = cv2.undistortPoints(
                pts_b.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

            pts_4d = cv2.triangulatePoints(a['P'], b['P'], pts_a_ud.T, pts_b_ud.T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T  # (N, 3), in base_link frame

            # sanity filter: reject points far outside a plausible
            # small-arm workspace (helps drop bad matches/outliers)
            valid = np.linalg.norm(pts_3d, axis=1) < max_dist
            if not np.any(valid):
                continue

            colors = []
            for (u, v) in pts_a[valid]:
                u_i, v_i = int(round(u)), int(round(v))
                if 0 <= v_i < a['img'].shape[0] and 0 <= u_i < a['img'].shape[1]:
                    b_, g_, r_ = a['img'][v_i, u_i]
                    colors.append([r_, g_, b_])
                else:
                    colors.append([128, 128, 128])

            all_points.append(pts_3d[valid])
            all_colors.append(np.array(colors))
            n_pairs_used += 1
            print(f'Pair ({i},{j}): {len(good)} matches, {len(pts_3d[valid])} valid points')

    if not all_points:
        print('No triangulated points -- check feature matches / poses.')
        return None

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)

    ply_path = out_dir / 'reconstruction.ply'
    write_ply(ply_path, points, colors)
    print(f'Reconstructed {len(points)} points from {n_pairs_used} image pairs -> {ply_path}')

    return str(ply_path)


if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='Directory containing images/ and manifest.yaml')
    parser.add_argument('--output-dir', default=None,
                         help='Where to write reconstruction.ply (default: dataset_dir)')
    parser.add_argument('--n-features', type=int, default=4000)
    parser.add_argument('--ratio-test', type=float, default=0.75)
    parser.add_argument('--min-matches', type=int, default=8)
    parser.add_argument('--max-dist', type=float, default=2.0)
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser()
    manifest_path = dataset_dir / 'manifest.yaml'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    ply_path = reconstruct_point_cloud(
        dataset_dir, manifest,
        n_features=args.n_features,
        ratio_test=args.ratio_test,
        min_matches=args.min_matches,
        max_dist=args.max_dist,
        out_dir=args.output_dir,
    )
    if ply_path is None:
        print('Reconstruction failed.')
    else:
        print(f'Done: {ply_path}')