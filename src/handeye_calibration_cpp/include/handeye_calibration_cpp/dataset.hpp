// dataset.yaml read/write. The Python nodes got this from PyYAML directly;
// in C++ the same layout is spelled out here so capture_node (writer) and
// calibration_node (reader) cannot disagree about it.
#ifndef HANDEYE_CALIBRATION_CPP__DATASET_HPP_
#define HANDEYE_CALIBRATION_CPP__DATASET_HPP_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "handeye_calibration_cpp/board_utils.hpp"
#include "handeye_calibration_cpp/solver.hpp"   // Sample, used by LoadedSample

namespace handeye_calibration_cpp
{

/// One entry of dataset.yaml: an image plus the gripper pose it was taken at.
/// The board pose is deliberately not stored -- every consumer re-detects it,
/// which is what lets a changed board spec or changed intrinsics actually take
/// effect.
struct DatasetEntry
{
  std::string image_file;
  cv::Matx33d R_gripper2base;
  cv::Vec3d t_gripper2base;
};

struct Dataset
{
  cv::Mat camera_matrix;              // 3x3 CV_64F
  cv::Mat dist_coeffs;                // 1xN CV_64F
  BoardConfig board;
  std::vector<DatasetEntry> entries;
};

/// Load <dataset_dir>/dataset.yaml. Throws std::runtime_error if it is missing
/// or malformed.
Dataset load_dataset(const std::string & dataset_dir);

/// Write <dataset_dir>/dataset.yaml, replacing any existing file.
void save_dataset(const std::string & dataset_dir, const Dataset & dataset);

/// Contents of handeye_result.yaml.
struct HandEyeResultFile
{
  std::string parent_frame;
  std::string child_frame;
  cv::Vec3d translation;
  cv::Vec4d rotation_xyzw;
  std::string method;
  int num_samples = 0;
  int num_skipped = 0;
};

void save_handeye_result(const std::string & path, const HandEyeResultFile & result);

/// A sample loaded from a dataset, carrying enough context to go back and look
/// at the capture it came from.
struct LoadedSample
{
  size_t index;              ///< entry number in dataset.yaml, not the position here
  std::string image_file;
  Sample sample;
};

/// Why an entry was dropped during load_samples.
struct SkippedEntry
{
  size_t index;
  std::string image_file;
  std::string reason;
};

/// Load a dataset and re-detect the board in every image.
///
/// `skipped` collects the entries that were dropped and why. The Python
/// calibration_node only counted them; keeping which ones is exactly the
/// information needed to go and look at the bad capture.
std::vector<LoadedSample> load_samples(
  const std::string & dataset_dir, const Dataset & dataset,
  std::vector<SkippedEntry> & skipped);

}  // namespace handeye_calibration_cpp

#endif  // HANDEYE_CALIBRATION_CPP__DATASET_HPP_
