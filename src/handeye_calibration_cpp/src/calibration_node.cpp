// calibration_node.cpp
//
// Exposes calibrate_dataset (handeye_calibration_interfaces/srv/CalibrateDataset):
// re-detects the board in every image of a dataset, solves the eye-in-hand
// AX=XB problem over the result, and writes handeye_result.yaml next to the
// dataset.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <handeye_calibration_interfaces/srv/calibrate_dataset.hpp>

#include "handeye_calibration_cpp/board_utils.hpp"
#include "handeye_calibration_cpp/dataset.hpp"
#include "handeye_calibration_cpp/solver.hpp"

namespace fs = std::filesystem;
using handeye_calibration_interfaces::srv::CalibrateDataset;
using namespace handeye_calibration_cpp;

class CalibrationNode : public rclcpp::Node
{
public:
  CalibrationNode()
  : Node("calibration_node")
  {
    // The solved transform is gripper -> camera; these name those two frames
    // in the result file. The camera frame changes with the camera, so it is
    // a parameter rather than a hard-coded frame.
    declare_parameter("parent_frame", "openarm_right_hand");
    declare_parameter("child_frame", "webcam_optical_frame");

    service_ = create_service<CalibrateDataset>(
      "calibrate_dataset",
      std::bind(
        &CalibrationNode::calibrate_callback, this, std::placeholders::_1,
        std::placeholders::_2));
    RCLCPP_INFO(get_logger(), "calibration_node ready, service: calibrate_dataset");
  }

private:
  void calibrate_callback(
    const std::shared_ptr<CalibrateDataset::Request> request,
    std::shared_ptr<CalibrateDataset::Response> response)
  {
    response->success = false;

    const std::string method =
      is_known_method(request->method) ? request->method : std::string("DANIILIDIS");

    Dataset dataset;
    try {
      dataset = load_dataset(request->dataset_dir);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "%s", e.what());
      return;
    }

    std::vector<SkippedEntry> skipped;
    std::vector<LoadedSample> loaded;
    try {
      loaded = load_samples(request->dataset_dir, dataset, skipped);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Could not load samples: %s", e.what());
      return;
    }

    const size_t n = loaded.size();
    RCLCPP_INFO(
      get_logger(), "Loaded %zu usable samples (%zu skipped)", n, skipped.size());
    for (const auto & s : skipped) {
      RCLCPP_WARN(
        get_logger(), "  skipped entry %zu (%s): %s", s.index, s.image_file.c_str(),
        s.reason.c_str());
    }
    if (n < 8) {
      RCLCPP_WARN(
        get_logger(), "Fewer than 8 samples -- solve will likely be poorly conditioned.");
    }
    if (n < 3) {
      return;
    }

    std::vector<Sample> samples;
    samples.reserve(n);
    for (const auto & l : loaded) {
      samples.push_back(l.sample);
    }

    HandEyeResult result;
    try {
      result = solve(samples, method);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Hand-eye solve failed: %s", e.what());
      return;
    }
    const cv::Vec4d quat = matrix_to_quat_xyzw(result.R);

    cv::Vec3d stddev(0.0, 0.0, 0.0);
    if (n >= 6) {
      stddev = leave_one_out_check(samples, method).stddev;
    }

    const std::string result_yaml_path =
      (fs::path(request->dataset_dir) / "handeye_result.yaml").string();

    HandEyeResultFile file;
    file.parent_frame = get_parameter("parent_frame").as_string();
    file.child_frame = get_parameter("child_frame").as_string();
    file.translation = result.t;
    file.rotation_xyzw = quat;
    file.method = method;
    file.num_samples = static_cast<int>(n);
    file.num_skipped = static_cast<int>(skipped.size());

    try {
      save_handeye_result(result_yaml_path, file);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Could not write result: %s", e.what());
      return;
    }

    response->success = true;
    response->result_yaml_path = result_yaml_path;
    response->translation = {result.t[0], result.t[1], result.t[2]};
    response->rotation_xyzw = {quat[0], quat[1], quat[2], quat[3]};
    response->stability_stddev = {stddev[0], stddev[1], stddev[2]};
  }

  rclcpp::Service<CalibrateDataset>::SharedPtr service_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CalibrationNode>());
  rclcpp::shutdown();
  return 0;
}
