#include "handeye_calibration_cpp/board_utils.hpp"

#include <map>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace handeye_calibration_cpp
{

namespace
{
const std::map<std::string, int> & dictionary_map()
{
  static const std::map<std::string, int> kMap = {
    {"4X4_50", cv::aruco::DICT_4X4_50},
    {"4X4_100", cv::aruco::DICT_4X4_100},
    {"4X4_250", cv::aruco::DICT_4X4_250},
    {"4X4_1000", cv::aruco::DICT_4X4_1000},
    {"5X5_50", cv::aruco::DICT_5X5_50},
    {"5X5_100", cv::aruco::DICT_5X5_100},
    {"5X5_250", cv::aruco::DICT_5X5_250},
    {"6X6_50", cv::aruco::DICT_6X6_50},
    {"6X6_250", cv::aruco::DICT_6X6_250},
    {"7X7_50", cv::aruco::DICT_7X7_50},
  };
  return kMap;
}
}  // namespace

std::vector<std::string> aruco_dictionary_names()
{
  std::vector<std::string> names;
  names.reserve(dictionary_map().size());
  for (const auto & [name, id] : dictionary_map()) {
    (void)id;
    names.push_back(name);
  }
  return names;
}

int aruco_dictionary_id(const std::string & name)
{
  const auto it = dictionary_map().find(name);
  if (it == dictionary_map().end()) {
    std::ostringstream oss;
    oss << "Unknown dictionary \"" << name << "\". Valid options: ";
    bool first = true;
    for (const auto & valid : aruco_dictionary_names()) {
      if (!first) {
        oss << ", ";
      }
      oss << valid;
      first = false;
    }
    throw std::invalid_argument(oss.str());
  }
  return it->second;
}

std::shared_ptr<CharucoTarget> build_board(
  int squares_x, int squares_y, double square_length_m, double marker_length_m,
  const std::string & dictionary_name, bool legacy_pattern)
{
  const cv::aruco::Dictionary aruco_dict =
    cv::aruco::getPredefinedDictionary(aruco_dictionary_id(dictionary_name));

  cv::aruco::CharucoBoard board(
    cv::Size(squares_x, squares_y), static_cast<float>(square_length_m),
    static_cast<float>(marker_length_m), aruco_dict);
  if (legacy_pattern) {
    board.setLegacyPattern(true);
  }

  auto target = std::make_shared<CharucoTarget>(
    CharucoTarget{
      board,
      cv::aruco::CharucoDetector(
        board, cv::aruco::CharucoParameters(), cv::aruco::DetectorParameters())});

  // The detector above was constructed from the local `board` copy. Rebind it
  // to the board that actually lives inside the returned struct so the two
  // cannot drift apart if the stored board is ever reconfigured.
  target->detector.setBoard(target->board);
  return target;
}

std::shared_ptr<CharucoTarget> build_board(const BoardConfig & cfg)
{
  return build_board(
    cfg.squares_x, cfg.squares_y, cfg.square_length_m, cfg.marker_length_m,
    cfg.dictionary, cfg.legacy_pattern);
}

std::optional<BoardPose> detect_board_pose(
  const cv::Mat & gray_image, CharucoTarget & target,
  const cv::Mat & camera_matrix, const cv::Mat & dist_coeffs, int min_corners)
{
  cv::Mat charuco_corners, charuco_ids;
  target.detector.detectBoard(gray_image, charuco_corners, charuco_ids);
  if (charuco_ids.empty() || charuco_ids.total() < static_cast<size_t>(min_corners)) {
    return std::nullopt;
  }

  cv::Mat obj_points, img_points;
  target.board.matchImagePoints(charuco_corners, charuco_ids, obj_points, img_points);
  if (obj_points.empty() || obj_points.total() < 4) {
    return std::nullopt;
  }

  cv::Vec3d rvec, tvec;
  const bool ok = cv::solvePnP(
    obj_points, img_points, camera_matrix, dist_coeffs, rvec, tvec, false,
    cv::SOLVEPNP_ITERATIVE);
  if (!ok) {
    return std::nullopt;
  }

  cv::Matx33d R;
  cv::Rodrigues(rvec, R);
  return BoardPose{R, tvec};
}

cv::Matx33d quat_to_matrix(double x, double y, double z, double w)
{
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  x /= n;
  y /= n;
  z /= n;
  w /= n;
  return cv::Matx33d(
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
}

cv::Vec4d matrix_to_quat_xyzw(const cv::Matx33d & R)
{
  // Shepperd's method: pick the branch whose divisor is largest, so the
  // division never runs close to zero.
  const double trace = R(0, 0) + R(1, 1) + R(2, 2);
  double x, y, z, w;

  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (R(2, 1) - R(1, 2)) / s;
    y = (R(0, 2) - R(2, 0)) / s;
    z = (R(1, 0) - R(0, 1)) / s;
  } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
    const double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
    w = (R(2, 1) - R(1, 2)) / s;
    x = 0.25 * s;
    y = (R(0, 1) + R(1, 0)) / s;
    z = (R(0, 2) + R(2, 0)) / s;
  } else if (R(1, 1) > R(2, 2)) {
    const double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
    w = (R(0, 2) - R(2, 0)) / s;
    x = (R(0, 1) + R(1, 0)) / s;
    y = 0.25 * s;
    z = (R(1, 2) + R(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
    w = (R(1, 0) - R(0, 1)) / s;
    x = (R(0, 2) + R(2, 0)) / s;
    y = (R(1, 2) + R(2, 1)) / s;
    z = 0.25 * s;
  }
  return cv::Vec4d(x, y, z, w);
}

}  // namespace handeye_calibration_cpp
