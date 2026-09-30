// Eye-in-hand AX=XB solve and its leave-one-out stability check -- the C++
// counterpart of handeye_calibration/solver.py.
#ifndef HANDEYE_CALIBRATION_CPP__SOLVER_HPP_
#define HANDEYE_CALIBRATION_CPP__SOLVER_HPP_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace handeye_calibration_cpp
{

/// gripper -> camera transform, as returned by cv::calibrateHandEye.
struct HandEyeResult
{
  cv::Matx33d R;
  cv::Vec3d t;
};

/// One capture: gripper pose in base coordinates, board pose in camera
/// coordinates.
struct Sample
{
  cv::Matx33d R_gripper2base;
  cv::Vec3d t_gripper2base;
  cv::Matx33d R_target2cam;
  cv::Vec3d t_target2cam;
};

/// Mean and per-axis standard deviation of the translations produced by the
/// leave-one-out subsets.
struct StabilityStats
{
  cv::Vec3d mean{0.0, 0.0, 0.0};
  cv::Vec3d stddev{0.0, 0.0, 0.0};
};

/// Method names accepted in CalibrateDataset.method, mapped to the
/// cv::HandEyeCalibrationMethod values.
///
/// ANDREFF is commented out in the Python and is likewise absent here: it
/// assumes a scale that this setup does not satisfy.
bool is_known_method(const std::string & name);
std::vector<std::string> method_names();

/// Solve for the gripper -> camera transform. Throws cv::Exception if the
/// solve is degenerate, and std::invalid_argument on an unknown method name.
HandEyeResult solve(const std::vector<Sample> & samples, const std::string & method_name);

/// Re-solve with each sample removed in turn and report how much the
/// translation moves. Large spread flags either an outlier sample (bad
/// detection, arm not settled) or a trajectory with insufficient rotation
/// diversity. Subsets whose solve throws are skipped, matching the Python.
StabilityStats leave_one_out_check(
  const std::vector<Sample> & samples, const std::string & method_name);

}  // namespace handeye_calibration_cpp

#endif  // HANDEYE_CALIBRATION_CPP__SOLVER_HPP_
