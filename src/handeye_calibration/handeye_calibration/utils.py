import os

import numpy as np
import cv2
import yaml
from cv2 import aruco


ARUCO_DICT_MAP = {
    '4X4_50': aruco.DICT_4X4_50, '4X4_100': aruco.DICT_4X4_100,
    '4X4_250': aruco.DICT_4X4_250, '4X4_1000': aruco.DICT_4X4_1000,
    '5X5_50': aruco.DICT_5X5_50, '5X5_100': aruco.DICT_5X5_100,
    '5X5_250': aruco.DICT_5X5_250,
    '6X6_50': aruco.DICT_6X6_50, '6X6_250': aruco.DICT_6X6_250,
    '7X7_50': aruco.DICT_7X7_50,
}


def build_board(squares_x, squares_y, square_length_m, marker_length_m,
                 dictionary_name, legacy_pattern=True):
    if dictionary_name not in ARUCO_DICT_MAP:
        raise ValueError(f'Unknown dictionary "{dictionary_name}". '
                          f'Valid options: {list(ARUCO_DICT_MAP.keys())}')
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dictionary_name])
    board = aruco.CharucoBoard((squares_x, squares_y), square_length_m,
                                marker_length_m, aruco_dict)
    if legacy_pattern:
        board.setLegacyPattern(True)
    detector_params = aruco.DetectorParameters()
    charuco_params = aruco.CharucoParameters()
    detector = aruco.CharucoDetector(board, charuco_params, detector_params)
    return board, detector



def detect_board_pose(gray_image, board, detector, camera_matrix, dist_coeffs,
                       min_corners=6):
    """Returns (R, t) -- 3x3 rotation matrix and 3-vector translation of the
    board expressed in the camera frame -- or None if detection failed."""
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray_image)
    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None

    # matchImagePoints pairs the corners that were actually detected with their
    # board-frame object points. getChessboardCorners() alone returns only the
    # full object-point array and knows nothing about this image, so solving
    # against it ignores the detection entirely.
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < 4:
        return None

    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix,
                                   dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten()


# def imgmsg_to_cv2(msg):
#     """Manual sensor_msgs/Image -> BGR numpy array conversion.
#     (No cv_bridge -- its compiled extension is linked against NumPy 1.x's
#     ABI and crashes under NumPy 2.x.)"""
#     encoding = msg.encoding
#     if encoding in ('bgr8', 'rgb8'):
#         channels = 3
#     elif encoding == 'mono8':
#         channels = 1
#     else:
#         raise ValueError(f"Unsupported image encoding '{encoding}'.")

#     buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
#     img = buf[:, :msg.width * channels]
#     img = img.reshape(msg.height, msg.width, channels) if channels > 1 \
#         else img.reshape(msg.height, msg.width)
#     if encoding == 'rgb8':
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     return img

def imgmsg_to_cv2(msg):
    """Manual sensor_msgs/Image -> numpy array conversion.
    (No cv_bridge -- its compiled extension is linked against NumPy 1.x's
    ABI and crashes under NumPy 2.x.)

    Color encodings (bgr8/rgb8/mono8) are returned as 3-channel BGR uint8.
    Depth encodings (16UC1/32FC1) are returned as raw single-channel arrays
    (uint16 millimetres / float32 metres respectively) -- no color
    conversion applies to depth data.
    """
    encoding = msg.encoding

    if encoding == '16UC1':
        buf = np.frombuffer(msg.data, dtype=np.uint16)
        step_elems = msg.step // 2  # step is in bytes; uint16 is 2 bytes/elem
        return buf.reshape(msg.height, step_elems)[:, :msg.width].copy()

    if encoding == '32FC1':
        buf = np.frombuffer(msg.data, dtype=np.float32)
        step_elems = msg.step // 4  # step is in bytes; float32 is 4 bytes/elem
        return buf.reshape(msg.height, step_elems)[:, :msg.width].copy()

    if encoding in ('bgr8', 'rgb8'):
        channels = 3
    elif encoding == 'mono8':
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding '{encoding}'.")

    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * channels]
    img = img.reshape(msg.height, msg.width, channels) if channels > 1 \
        else img.reshape(msg.height, msg.width)
    if encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == 'mono8':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_dataset(dataset_dir):
    """Load a capture_node dataset and re-detect the board in every image.

    Board poses are deliberately not stored by capture_node, so every consumer
    re-detects them; that is what lets a changed board spec or changed
    intrinsics actually take effect.

    Returns (samples, meta). ``samples`` is a list of dicts, one per entry whose
    board was found, each carrying:

        index        entry number in dataset.yaml (NOT the position in this list)
        image_file   the file it came from
        R_g2b, t_g2b gripper pose in base coordinates
        R_t2c, t_t2c board pose in camera coordinates

    ``meta`` carries camera_matrix, dist_coeffs, the board config, and
    ``skipped`` -- a list of (index, image_file, reason) for entries that were
    dropped. The existing loaders count skips but discard which ones, which is
    exactly the information needed to go and look at the bad capture.
    """
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

    samples, skipped = [], []
    for index, entry in enumerate(dataset['entries']):
        image_file = entry['image_file']
        image_path = os.path.join(dataset_dir, 'images', image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append((index, image_file, 'image unreadable'))
            continue

        result = detect_board_pose(img, board, detector, camera_matrix,
                                    dist_coeffs, min_corners)
        if result is None:
            skipped.append((index, image_file, 'board not detected'))
            continue

        R_t2c, t_t2c = result
        samples.append({
            'index': index,
            'image_file': image_file,
            'R_g2b': np.array(entry['R_gripper2base']),
            't_g2b': np.array(entry['t_gripper2base']),
            'R_t2c': R_t2c,
            't_t2c': t_t2c,
        })

    meta = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'board': board_cfg,
        'skipped': skipped,
        'total_entries': len(dataset['entries']),
    }
    return samples, meta


def quat_to_matrix(x, y, z, w):
    """Normalized quaternion -> 3x3 rotation matrix."""
    n = np.array([x, y, z, w])
    n = n / np.linalg.norm(n)
    x, y, z, w = n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])