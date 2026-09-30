// auto_calibration_node.cpp
//
// Drives the calibration pipeline for a preset number of samples.
//
// Per sample: call the point sampler's ~/move_to_next_point to reach the next
// pose, then capture_node's ~/sample_position to record the frame and the TF
// lookup. Once all samples are collected, call calibrate_dataset to solve
// hand-eye over the result.
//
// The sampler is whatever exposes ~/move_to_next_point -- run_points_node
// replaying a recorded manifest (the openarm setup), or motor_bridge prompting
// for each pose by hand (the SO-101 setup).

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include <handeye_calibration_interfaces/srv/calibrate_dataset.hpp>
#include <handeye_calibration_interfaces/srv/move_to_next_point.hpp>
#include <handeye_calibration_interfaces/srv/sample_position.hpp>

using handeye_calibration_interfaces::srv::CalibrateDataset;
using handeye_calibration_interfaces::srv::MoveToNextPoint;
using handeye_calibration_interfaces::srv::SamplePosition;

class CalibrationOrchestrator : public rclcpp::Node
{
public:
  CalibrationOrchestrator()
  : Node("auto_calibration_node")
  {
    declare_parameter("output_dir", "handeye_dataset");
    declare_parameter("calibration_method", "DANIILIDIS");
    declare_parameter("num_samples", 200);
    declare_parameter("point_sampler_node", "/run_points_node");
    declare_parameter("capture_node", "/capture_node");

    const std::string capture_ns = get_parameter("capture_node").as_string();
    sample_client_ = create_client<SamplePosition>(capture_ns + "/sample_position");
    calibrate_client_ = create_client<CalibrateDataset>("calibrate_dataset");

    const std::string mover_ns = get_parameter("point_sampler_node").as_string();
    move_client_ = create_client<MoveToNextPoint>(mover_ns + "/move_to_next_point");
  }

  bool run()
  {
    const std::string output_dir = get_parameter("output_dir").as_string();
    const int num_samples = static_cast<int>(get_parameter("num_samples").as_int());

    const int num_captured = collect_dataset(output_dir, num_samples);
    if (num_captured == 0) {
      RCLCPP_ERROR(get_logger(), "Dataset collection failed (0 samples captured).");
      return false;
    }
    RCLCPP_INFO(
      get_logger(), "Captured %d viewpoints to %s.", num_captured, output_dir.c_str());

    RCLCPP_INFO(get_logger(), "Waiting for calibrate_dataset service...");
    calibrate_client_->wait_for_service();

    auto calib_request = std::make_shared<CalibrateDataset::Request>();
    calib_request->dataset_dir = output_dir;
    calib_request->method = get_parameter("calibration_method").as_string();

    RCLCPP_INFO(get_logger(), "Calling CalibrateDataset...");
    auto future = calibrate_client_->async_send_request(calib_request);
    if (future.wait_for(std::chrono::seconds(600)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "calibrate_dataset call timed out.");
      return false;
    }
    const auto response = future.get();

    if (!response->success) {
      RCLCPP_ERROR(get_logger(), "Calibration solve failed.");
      return false;
    }

    RCLCPP_INFO(
      get_logger(), "Calibration complete. Result written to %s",
      response->result_yaml_path.c_str());
    RCLCPP_INFO(
      get_logger(), "  translation: [%.6f, %.6f, %.6f]", response->translation[0],
      response->translation[1], response->translation[2]);
    RCLCPP_INFO(
      get_logger(), "  rotation_xyzw: [%.6f, %.6f, %.6f, %.6f]", response->rotation_xyzw[0],
      response->rotation_xyzw[1], response->rotation_xyzw[2], response->rotation_xyzw[3]);
    RCLCPP_INFO(
      get_logger(), "  leave-one-out stddev: [%.6f, %.6f, %.6f]",
      response->stability_stddev[0], response->stability_stddev[1],
      response->stability_stddev[2]);
    return true;
  }

private:
  /// Call a service and return its response, or nullptr.
  template<typename ClientT, typename RequestT>
  auto call(const ClientT & client, const RequestT & request)
  {
    auto future = client->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(120)) != std::future_status::ready) {
      return decltype(future.get())(nullptr);
    }
    return future.get();
  }

  int collect_dataset(const std::string & output_dir, int num_samples)
  {
    RCLCPP_INFO(get_logger(), "Waiting for move_to_next_point service...");
    move_client_->wait_for_service();
    RCLCPP_INFO(get_logger(), "Waiting for sample_position service...");
    sample_client_->wait_for_service();

    int num_captured = 0;
    for (int i = 1; i <= num_samples && rclcpp::ok(); ++i) {
      RCLCPP_INFO(get_logger(), "[%d/%d] Calling move_to_next_point...", i, num_samples);
      const auto move_response =
        call(move_client_, std::make_shared<MoveToNextPoint::Request>());

      if (!move_response) {
        RCLCPP_ERROR(
          get_logger(), "Sample %d: move_to_next_point call failed (no response).", i);
        continue;
      }
      if (!move_response->success) {
        RCLCPP_ERROR(get_logger(), "Sample %d: %s", i, move_response->message.c_str());
        continue;
      }

      RCLCPP_INFO(get_logger(), "[%d/%d] Calling sample_position...", i, num_samples);
      auto sample_request = std::make_shared<SamplePosition::Request>();
      sample_request->dataset_dir = output_dir;
      const auto sample_response = call(sample_client_, sample_request);

      if (!sample_response) {
        RCLCPP_ERROR(
          get_logger(), "Sample %d: sample_position call failed (no response).", i);
        continue;
      }
      if (!sample_response->success) {
        RCLCPP_ERROR(get_logger(), "Sample %d: %s", i, sample_response->message.c_str());
        continue;
      }

      ++num_captured;
      RCLCPP_INFO(get_logger(), "Sample %d: %s", i, sample_response->message.c_str());
    }

    RCLCPP_INFO(
      get_logger(),
      "Dataset collection finished: %d/%d usable samples, dataset written to %s",
      num_captured, num_samples, output_dir.c_str());
    return num_captured;
  }

  rclcpp::Client<SamplePosition>::SharedPtr sample_client_;
  rclcpp::Client<CalibrateDataset>::SharedPtr calibrate_client_;
  rclcpp::Client<MoveToNextPoint>::SharedPtr move_client_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CalibrationOrchestrator>();

  // run() blocks on service futures, so the executor has to spin elsewhere --
  // the Python got this from spin_until_future_complete re-entering the same
  // loop, which has no direct C++ equivalent.
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() {executor.spin();});

  const bool success = node->run();

  executor.cancel();
  spin_thread.join();
  rclcpp::shutdown();
  return success ? 0 : 1;
}
