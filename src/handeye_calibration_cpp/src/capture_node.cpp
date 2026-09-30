// capture_node.cpp
//
// Exposes ~/sample_position (handeye_calibration_interfaces/srv/SamplePosition).
//
// Each call captures the current camera frame + gripper pose (looked up via TF)
// and appends it as a new entry to {request.dataset_dir}/dataset.yaml, writing
// the image to {request.dataset_dir}/images/. The dataset's camera_matrix,
// dist_coeffs and board config are populated on the first sample for a given
// dataset_dir.
//
// Assumes the robot is already stationary at the desired pose when this service
// is called -- moving it there is run_points_node's job (~/move_to_next_point).

#include <filesystem>
#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <handeye_calibration_interfaces/srv/sample_position.hpp>

#include "handeye_calibration_cpp/board_utils.hpp"
#include "handeye_calibration_cpp/dataset.hpp"

namespace fs = std::filesystem;
using handeye_calibration_interfaces::srv::SamplePosition;
using namespace handeye_calibration_cpp;

class CaptureNode : public rclcpp::Node
{
public:
  CaptureNode()
  : Node("capture_node")
  {
    declare_parameter("base_frame", "base_link");
    declare_parameter("gripper_frame", "gripper_link");
    declare_parameter("image_topic", "/camera/color/image_raw");
    declare_parameter("camera_info_topic", "/camera/color/camera_info");

    // -- board config, passed through into dataset.yaml for calibration_node --
    declare_parameter("squares_x", 11);
    declare_parameter("squares_y", 8);
    declare_parameter("square_length_m", 0.015);
    declare_parameter("marker_length_m", 0.011);
    declare_parameter("dictionary", "4X4_50");
    declare_parameter("legacy_pattern", true);
    declare_parameter("min_charuco_corners", 6);

    base_frame_ = get_parameter("base_frame").as_string();
    gripper_frame_ = get_parameter("gripper_frame").as_string();

    caminfo_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      get_parameter("camera_info_topic").as_string(), 10,
      std::bind(&CaptureNode::caminfo_cb, this, std::placeholders::_1));
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      get_parameter("image_topic").as_string(), 10,
      std::bind(&CaptureNode::image_cb, this, std::placeholders::_1));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    service_ = create_service<SamplePosition>(
      "~/sample_position",
      std::bind(
        &CaptureNode::srv_sample_position, this, std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "capture_node ready. Service: ~/sample_position");
  }

private:
  // ------------------------------------------------------------
  // Camera / TF helpers
  // ------------------------------------------------------------
  void caminfo_cb(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (msg->k[0] == 0.0 || msg->k[4] == 0.0) {
      // A webcam has no factory intrinsics: until camera_info_url points at
      // a calibrated file, usb_cam publishes an all-zero CameraInfo. Keep
      // the subscription alive so a later valid message still lands.
      RCLCPP_WARN_ONCE(
        get_logger(),
        "CameraInfo has zero focal length -- camera is uncalibrated. "
        "Run intrinsics.launch.py first.");
      return;
    }

    cv::Mat k(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
      k.at<double>(i / 3, i % 3) = msg->k[i];
    }
    camera_matrix_ = k;

    cv::Mat d(1, static_cast<int>(msg->d.size()), CV_64F);
    for (size_t i = 0; i < msg->d.size(); ++i) {
      d.at<double>(0, static_cast<int>(i)) = msg->d[i];
    }
    dist_coeffs_ = d;

    caminfo_sub_.reset();
  }

  void image_cb(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      latest_bgr_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN(get_logger(), "cv_bridge conversion failed: %s", e.what());
    }
  }

  /// camera_matrix_ is set by caminfo_cb, processed by the same executor
  /// during the idle time between service calls -- no waiting here, this is a
  /// single check against whatever's already arrived.
  bool has_camera_intrinsics() const {return !camera_matrix_.empty();}

  /// Pose of gripper_frame expressed in base_frame, from whatever the TF
  /// buffer already has -- single attempt, no retrying.
  bool get_base_to_gripper(cv::Matx33d & R, cv::Vec3d & t)
  {
    try {
      const auto tf = tf_buffer_->lookupTransform(
        base_frame_, gripper_frame_, tf2::TimePointZero);
      const auto & q = tf.transform.rotation;
      const auto & v = tf.transform.translation;
      R = quat_to_matrix(q.x, q.y, q.z, q.w);
      t = cv::Vec3d(v.x, v.y, v.z);
      return true;
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(get_logger(), "TF lookup failed: %s", e.what());
      return false;
    }
  }

  // ------------------------------------------------------------
  /// Load the dataset if it already exists, otherwise start a fresh one
  /// seeded with the current intrinsics and this node's board parameters.
  Dataset load_or_init_dataset(const std::string & dataset_dir)
  {
    if (fs::exists(fs::path(dataset_dir) / "dataset.yaml")) {
      return load_dataset(dataset_dir);
    }

    Dataset dataset;
    dataset.camera_matrix = camera_matrix_;
    dataset.dist_coeffs = dist_coeffs_;
    dataset.board.squares_x = get_parameter("squares_x").as_int();
    dataset.board.squares_y = get_parameter("squares_y").as_int();
    dataset.board.square_length_m = get_parameter("square_length_m").as_double();
    dataset.board.marker_length_m = get_parameter("marker_length_m").as_double();
    dataset.board.dictionary = get_parameter("dictionary").as_string();
    dataset.board.legacy_pattern = get_parameter("legacy_pattern").as_bool();
    dataset.board.min_charuco_corners = get_parameter("min_charuco_corners").as_int();
    return dataset;
  }

  // ------------------------------------------------------------
  void srv_sample_position(
    const std::shared_ptr<SamplePosition::Request> request,
    std::shared_ptr<SamplePosition::Response> response)
  {
    const fs::path out_dir(request->dataset_dir);
    const fs::path images_dir = out_dir / "images";

    if (!has_camera_intrinsics()) {
      response->success = false;
      response->message =
        "No valid camera intrinsics received yet (zero focal length means uncalibrated).";
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    if (latest_bgr_.empty()) {
      response->success = false;
      response->message = "No camera frame received yet.";
      RCLCPP_WARN(get_logger(), "%s", response->message.c_str());
      return;
    }
    const cv::Mat bgr = latest_bgr_.clone();

    cv::Matx33d R_gripper2base;
    cv::Vec3d t_gripper2base;
    if (!get_base_to_gripper(R_gripper2base, t_gripper2base)) {
      response->success = false;
      response->message = "TF lookup failed.";
      RCLCPP_WARN(get_logger(), "%s", response->message.c_str());
      return;
    }

    Dataset dataset;
    try {
      dataset = load_or_init_dataset(request->dataset_dir);
    } catch (const std::exception & e) {
      response->success = false;
      response->message = std::string("Could not load dataset: ") + e.what();
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    const size_t idx = dataset.entries.size();
    char image_file[32];
    std::snprintf(image_file, sizeof(image_file), "sample_%03zu.png", idx);

    try {
      fs::create_directories(images_dir);
      if (!cv::imwrite((images_dir / image_file).string(), bgr)) {
        throw std::runtime_error("cv::imwrite returned false");
      }
      dataset.entries.push_back(DatasetEntry{image_file, R_gripper2base, t_gripper2base});
      save_dataset(request->dataset_dir, dataset);
    } catch (const std::exception & e) {
      response->success = false;
      response->message = std::string("Could not write sample: ") + e.what();
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    RCLCPP_INFO(get_logger(), "Saved sample %zu -> images/%s", idx, image_file);

    response->success = true;
    response->message = "Saved sample " + std::to_string(idx);
    response->image_file = image_file;
    response->sample_index = static_cast<int32_t>(idx);
  }

  std::string base_frame_;
  std::string gripper_frame_;

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Mat latest_bgr_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Service<SamplePosition>::SharedPtr service_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CaptureNode>());
  rclcpp::shutdown();
  return 0;
}
