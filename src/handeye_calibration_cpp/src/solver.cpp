#include "handeye_calibration_cpp/solver.hpp"

#include <cmath>
#include <map>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace handeye_calibration_cpp
{

namespace
{
const std::map<std::string, cv::HandEyeCalibrationMethod> & method_map()
{
  static const std::map<std::string, cv::HandEyeCalibrationMethod> kMap = {
    {"TSAI", cv::CALIB_HAND_EYE_TSAI},
    {"PARK", cv::CALIB_HAND_EYE_PARK},
    {"HORAUD", cv::CALIB_HAND_EYE_HORAUD},
    {"DANIILIDIS", cv::CALIB_HAND_EYE_DANIILIDIS},
  };
  return kMap;
}

void split(
  const std::vector<Sample> & samples,
  std::vector<cv::Mat> & R_g2b, std::vector<cv::Mat> & t_g2b,
  std::vector<cv::Mat> & R_t2c, std::vector<cv::Mat> & t_t2c)
{
  R_g2b.reserve(samples.size());
  t_g2b.reserve(samples.size());
  R_t2c.reserve(samples.size());
  t_t2c.reserve(samples.size());
  for (const auto & s : samples) {
    R_g2b.emplace_back(cv::Mat(s.R_gripper2base));
    t_g2b.emplace_back(cv::Mat(s.t_gripper2base));
    R_t2c.emplace_back(cv::Mat(s.R_target2cam));
    t_t2c.emplace_back(cv::Mat(s.t_target2cam));
  }
}
}  // namespace

bool is_known_method(const std::string & name)
{
  return method_map().count(name) > 0;
}

std::vector<std::string> method_names()
{
  std::vector<std::string> names;
  names.reserve(method_map().size());
  for (const auto & [name, id] : method_map()) {
    (void)id;
    names.push_back(name);
  }
  return names;
}

HandEyeResult solve(const std::vector<Sample> & samples, const std::string & method_name)
{
  const auto it = method_map().find(method_name);
  if (it == method_map().end()) {
    throw std::invalid_argument("Unknown hand-eye method \"" + method_name + "\".");
  }

  std::vector<cv::Mat> R_g2b, t_g2b, R_t2c, t_t2c;
  split(samples, R_g2b, t_g2b, R_t2c, t_t2c);

  cv::Mat R_cam2gripper, t_cam2gripper;
  cv::calibrateHandEye(
    R_g2b, t_g2b, R_t2c, t_t2c, R_cam2gripper, t_cam2gripper, it->second);

  // calibrateHandEye returns CV_64F here, but the conversion is made explicit
  // so a differently-typed output could never be reinterpreted bit-for-bit.
  cv::Mat R64, t64;
  R_cam2gripper.reshape(1, 3).convertTo(R64, CV_64F);
  t_cam2gripper.reshape(1, 3).convertTo(t64, CV_64F);

  HandEyeResult result;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      result.R(r, c) = R64.at<double>(r, c);
    }
  }
  result.t = cv::Vec3d(
    t64.at<double>(0, 0), t64.at<double>(1, 0), t64.at<double>(2, 0));
  return result;
}

StabilityStats leave_one_out_check(
  const std::vector<Sample> & samples, const std::string & method_name)
{
  const size_t n = samples.size();
  std::vector<cv::Vec3d> translations;
  translations.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    std::vector<Sample> subset;
    subset.reserve(n - 1);
    for (size_t j = 0; j < n; ++j) {
      if (j != i) {
        subset.push_back(samples[j]);
      }
    }
    try {
      translations.push_back(solve(subset, method_name).t);
    } catch (const cv::Exception &) {
      continue;
    }
  }

  StabilityStats stats;
  if (translations.empty()) {
    return stats;
  }

  for (const auto & t : translations) {
    stats.mean += t;
  }
  stats.mean *= 1.0 / static_cast<double>(translations.size());

  cv::Vec3d variance(0.0, 0.0, 0.0);
  for (const auto & t : translations) {
    const cv::Vec3d d = t - stats.mean;
    variance += cv::Vec3d(d[0] * d[0], d[1] * d[1], d[2] * d[2]);
  }
  variance *= 1.0 / static_cast<double>(translations.size());  // population std, as numpy
  stats.stddev = cv::Vec3d(
    std::sqrt(variance[0]), std::sqrt(variance[1]), std::sqrt(variance[2]));
  return stats;
}

}  // namespace handeye_calibration_cpp
