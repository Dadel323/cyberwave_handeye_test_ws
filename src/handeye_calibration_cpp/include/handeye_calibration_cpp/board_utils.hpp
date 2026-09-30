// Board construction and ChArUco pose detection, shared by capture_node,
// calibration_node and intrinsics_node -- the C++ counterpart of
// handeye_calibration/utils.py.
#ifndef HANDEYE_CALIBRATION_CPP__BOARD_UTILS_HPP_
#define HANDEYE_CALIBRATION_CPP__BOARD_UTILS_HPP_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

namespace handeye_calibration_cpp
{

/// Board spec as it is stored in dataset.yaml.
struct BoardConfig
{
  int squares_x = 11;
  int squares_y = 8;
  double square_length_m = 0.015;
  double marker_length_m = 0.011;
  std::string dictionary = "4X4_50";
  bool legacy_pattern = true;
  int min_charuco_corners = 6;
};

/// A CharucoBoard together with the detector built over it. The detector holds
/// a reference to the board, so the two are kept in one object rather than
/// returned as a loose pair.
struct CharucoTarget
{
  cv::aruco::CharucoBoard board;
  cv::aruco::CharucoDetector detector;
};

/// Board pose in the camera frame.
struct BoardPose
{
  cv::Matx33d R;
  cv::Vec3d t;
};

/// Maps the dictionary names used in dataset.yaml onto cv::aruco predefined
/// dictionary ids. Throws std::invalid_argument on an unknown name.
int aruco_dictionary_id(const std::string & name);

/// Valid dictionary names, for error messages.
std::vector<std::string> aruco_dictionary_names();

/// Build the board and its detector. Throws std::invalid_argument if the
/// dictionary name is not recognised.
std::shared_ptr<CharucoTarget> build_board(
  int squares_x, int squares_y, double square_length_m, double marker_length_m,
  const std::string & dictionary_name, bool legacy_pattern = true);

std::shared_ptr<CharucoTarget> build_board(const BoardConfig & cfg);

/// Board pose expressed in the camera frame, or nullopt if detection failed.
///
/// matchImagePoints pairs the corners that were actually detected with their
/// board-frame object points. getChessboardCorners() alone returns only the
/// full object-point array and knows nothing about this image, so solving
/// against it ignores the detection entirely.
std::optional<BoardPose> detect_board_pose(
  const cv::Mat & gray_image, CharucoTarget & target,
  const cv::Mat & camera_matrix, const cv::Mat & dist_coeffs,
  int min_corners = 6);

/// Normalized quaternion -> 3x3 rotation matrix.
cv::Matx33d quat_to_matrix(double x, double y, double z, double w);

/// 3x3 rotation matrix -> quaternion, in (x, y, z, w) order.
///
/// The Python side went through transforms3d, whose mat2quat returns
/// (w, x, y, z); reading that positionally as xyzw shifted every component by
/// one and rotated the camera frame by ~180 deg in RViz. Returning a fixed
/// xyzw order here is what keeps that from recurring.
cv::Vec4d matrix_to_quat_xyzw(const cv::Matx33d & R);

}  // namespace handeye_calibration_cpp

#endif  // HANDEYE_CALIBRATION_CPP__BOARD_UTILS_HPP_
