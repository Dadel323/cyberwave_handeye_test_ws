// run_points_node.cpp
//
// Exposes ~/move_to_next_point (handeye_calibration_interfaces/srv/MoveToNextPoint).
//
// Each call advances to the NEXT joint-space pose loaded from manifest.yaml and
// moves the robot there (one pose per call).
//
// Bypasses MoveIt/IK entirely -- publishes straight to the commands topic,
// which motor_bridge already subscribes to and drives via its 50 Hz loop.
// Requires motor_bridge to already be running with torque enabled (its normal
// resting state whenever a move_to_next_point call isn't active).
//
// Usage:
//     ros2 run handeye_calibration_cpp run_points_node --ros-args -p
//     recorded_points_path:=/path/to/record_positions/data/manifest.yaml
//
// Then, from another terminal:
//
//     ros2 service call /run_points_node/move_to_next_point
//     handeye_calibration_interfaces/srv/MoveToNextPoint "{}"

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <yaml-cpp/yaml.h>

#include <handeye_calibration_interfaces/srv/move_to_next_point.hpp>

using handeye_calibration_interfaces::srv::MoveToNextPoint;
using sensor_msgs::msg::JointState;
using trajectory_msgs::msg::JointTrajectory;
using trajectory_msgs::msg::JointTrajectoryPoint;

/// A recorded pose: joint name -> position. std::map keeps a stable ordering,
/// so the names and positions published always line up.
using JointPose = std::map<std::string, double>;

class RunPointsNode : public rclcpp::Node
{
public:
  RunPointsNode()
  : Node("run_points_node")
  {
    declare_parameter(
      "recorded_points_path",
      "/home/cyb/projects/ros_ws/src/record_positions/data/manifest.yaml");
    declare_parameter("commands_topic", "/so101_follower/joint_commands");
    declare_parameter("joint_state_topic", "/so101_follower/joint_states");
    declare_parameter("use_joint_command", false);
    declare_parameter("trajectory_duration_sec", 1.5);

    const std::string joint_state_topic = get_parameter("joint_state_topic").as_string();
    const std::string recorded_points_path =
      get_parameter("recorded_points_path").as_string();
    const std::string topic = get_parameter("commands_topic").as_string();
    use_joint_command_ = get_parameter("use_joint_command").as_bool();
    trajectory_duration_sec_ = get_parameter("trajectory_duration_sec").as_double();

    if (!recorded_points_path.empty()) {
      poses_ = load_joint_poses(recorded_points_path);
      RCLCPP_INFO(
        get_logger(), "Loaded %zu joint-space poses from %s", poses_.size(),
        recorded_points_path.c_str());
    } else {
      RCLCPP_WARN(
        get_logger(),
        "No recorded_points_path given -- move_to_next_point calls will have "
        "nothing to advance through.");
    }

    if (!use_joint_command_) {
      traj_pub_ = create_publisher<JointTrajectory>(topic, 10);
    } else {
      pub_ = create_publisher<JointState>(topic, 10);
    }

    service_ = create_service<MoveToNextPoint>(
      "~/move_to_next_point",
      std::bind(
        &RunPointsNode::srv_move_to_next_point, this, std::placeholders::_1,
        std::placeholders::_2));

    joint_state_sub_ = create_subscription<JointState>(
      joint_state_topic, 10,
      std::bind(&RunPointsNode::joint_state_cb, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "run_points_node ready. Service: ~/move_to_next_point");
  }

private:
  void joint_state_cb(const JointState::SharedPtr msg)
  {
    joint_states_.clear();
    const size_t n = std::min(msg->name.size(), msg->position.size());
    for (size_t i = 0; i < n; ++i) {
      joint_states_[msg->name[i]] = msg->position[i];
    }
  }

  std::vector<JointPose> load_joint_poses(const std::string & path)
  {
    std::vector<JointPose> poses;
    YAML::Node manifest;
    try {
      manifest = YAML::LoadFile(path);
    } catch (const YAML::Exception & e) {
      RCLCPP_ERROR(get_logger(), "Could not read %s: %s", path.c_str(), e.what());
      return poses;
    }

    const YAML::Node entries = manifest["entries"];
    if (!entries || !entries.IsSequence()) {
      return poses;
    }

    size_t skipped = 0;
    for (const auto & entry : entries) {
      const YAML::Node joint_positions = entry["joint_positions"];
      if (!joint_positions || !joint_positions.IsMap() || joint_positions.size() == 0) {
        ++skipped;
        continue;
      }
      JointPose pose;
      for (const auto & kv : joint_positions) {
        pose[kv.first.as<std::string>()] = kv.second.as<double>();
      }
      poses.push_back(pose);
    }

    if (skipped > 0) {
      RCLCPP_WARN(
        get_logger(), "%zu entries had no joint_positions recorded, skipped.", skipped);
    }
    return poses;
  }

  void publish_joint_positions(const JointPose & pose)
  {
    JointState msg;
    msg.header.stamp = now();
    for (const auto & [name, position] : pose) {
      msg.name.push_back(name);
      msg.position.push_back(position);
    }
    pub_->publish(msg);
    // Block until the arm has arrived, exactly as the trajectory path below
    // does. Without this the service returns while the joints are still
    // travelling, and capture_node -- which is called the moment it returns --
    // pairs a motion-blurred frame with a TF pose from the wrong instant.
    // Nothing errors; the samples are simply wrong.
    sleep_sec(trajectory_duration_sec_ + 1.0);
  }

  void publish_joint_trajectory(const JointPose & pose)
  {
    JointTrajectory msg;
    JointTrajectoryPoint point;
    for (const auto & [name, position] : pose) {
      msg.joint_names.push_back(name);
      point.positions.push_back(position);
    }

    const double duration_sec = trajectory_duration_sec_;
    point.time_from_start.sec = static_cast<int32_t>(duration_sec);
    point.time_from_start.nanosec =
      static_cast<uint32_t>((duration_sec - static_cast<int32_t>(duration_sec)) * 1e9);
    msg.points.push_back(point);

    traj_pub_->publish(msg);
    sleep_sec(duration_sec + 1.0);  // Wait for the trajectory to complete
  }

  static void sleep_sec(double seconds)
  {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  }

  // ------------------------------------------------------------
  void srv_move_to_next_point(
    const std::shared_ptr<MoveToNextPoint::Request> /*request*/,
    std::shared_ptr<MoveToNextPoint::Response> response)
  {
    if (next_index_ >= poses_.size()) {
      response->success = false;
      response->message = "No more recorded poses (" + std::to_string(poses_.size()) +
        " total, all used).";
      RCLCPP_WARN(get_logger(), "%s", response->message.c_str());
      return;
    }

    const JointPose & pose = poses_[next_index_];
    ++next_index_;
    RCLCPP_INFO(
      get_logger(), "[recorded %zu/%zu] Moving to %zu joints", next_index_, poses_.size(),
      pose.size());

    if (!use_joint_command_) {
      publish_joint_trajectory(pose);
    } else {
      publish_joint_positions(pose);
    }

    response->success = true;
    response->message = "Moved to recorded pose";
    for (const auto & [name, position] : pose) {
      response->joint_names.push_back(name);
      response->joint_positions.push_back(position);
    }
  }

  std::vector<JointPose> poses_;
  size_t next_index_ = 0;
  bool use_joint_command_ = false;
  double trajectory_duration_sec_ = 1.5;
  std::map<std::string, double> joint_states_;

  rclcpp::Publisher<JointTrajectory>::SharedPtr traj_pub_;
  rclcpp::Publisher<JointState>::SharedPtr pub_;
  rclcpp::Subscription<JointState>::SharedPtr joint_state_sub_;
  rclcpp::Service<MoveToNextPoint>::SharedPtr service_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RunPointsNode>();

  // The service handler sleeps while the arm travels. On a single-threaded
  // executor that would also stall the joint_states subscription; a
  // multi-threaded executor keeps callbacks flowing during the move.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
