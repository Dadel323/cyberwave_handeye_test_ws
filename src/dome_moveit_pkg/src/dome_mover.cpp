#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <moveit_msgs/srv/get_cartesian_path.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <handeye_calibration_interfaces/srv/move_to_next_point.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;
using MoveToNextPoint = handeye_calibration_interfaces::srv::MoveToNextPoint;

class DomeMover : public rclcpp::Node
{
public:
    DomeMover()
        : Node("dome_mover"), gen_(rd_())
    {
        // -- dome geometry --
        this->declare_parameter("radius", 0.2);
        this->declare_parameter("center_x", -0.18);
        this->declare_parameter("center_y", 0.34);
        this->declare_parameter("center_z", 0.2);
        this->declare_parameter("max_attempts", 50);

        // -- MoveIt / IK --
        this->declare_parameter("planning_group", "right_arm");
        this->declare_parameter("end_effector_link", "openarm_right_hand");
        this->declare_parameter("base_frame", "world");
        this->declare_parameter("joints_to_keep", std::vector<std::string>{
                                                      "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
                                                      "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
                                                      "openarm_right_joint7"});
        this->declare_parameter("commands_topic", "/right_joint_trajectory_controller/joint_trajectory");

        // -- attached camera collision object --
        this->declare_parameter("attach_camera", true);
        this->declare_parameter("camera_link_name", "openarm_right_hand");
        this->declare_parameter("camera_box_size", std::vector<double>{0.09, 0.025, 0.025});
        this->declare_parameter("camera_mount_xyz", std::vector<double>{});
        this->declare_parameter("camera_mount_qxyzw", std::vector<double>{});

        radius_ = this->get_parameter("radius").as_double();
        center_x_ = this->get_parameter("center_x").as_double();
        center_y_ = this->get_parameter("center_y").as_double();
        center_z_ = this->get_parameter("center_z").as_double();
        max_attempts_ = this->get_parameter("max_attempts").as_int();

        planning_group_ = this->get_parameter("planning_group").as_string();
        end_effector_link_ = this->get_parameter("end_effector_link").as_string();
        base_frame_ = this->get_parameter("base_frame").as_string();
        joints_to_keep_ = this->get_parameter("joints_to_keep").as_string_array();
        std::string commands_topic = this->get_parameter("commands_topic").as_string();

        client_node_ = std::make_shared<rclcpp::Node>("dome_mover_client");
        ik_client_ = client_node_->create_client<moveit_msgs::srv::GetPositionIK>("/compute_ik");
        plan_client_ = client_node_->create_client<moveit_msgs::srv::GetCartesianPath>("/compute_cartesian_path");
        scene_client_ = client_node_->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");

        traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(commands_topic, 10);

        if (this->get_parameter("attach_camera").as_bool())
        {
            attach_camera_object();
        }

        service_ = this->create_service<MoveToNextPoint>(
            "~/move_to_next_point",
            std::bind(&DomeMover::handle_move_to_next_point, this,
                      std::placeholders::_1, std::placeholders::_2));

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/dome_marker", 10);
        marker_timer_ = this->create_wall_timer(1s, std::bind(&DomeMover::publish_dome_marker, this));

        RCLCPP_INFO(this->get_logger(), "dome_mover ready. Service: ~/move_to_next_point");
    }

private:
    // ------------------------------------------------------------
    // Dome geometry visualization -- a ring at the dome's equator
    // (radius_, centered at center_x_/y_/z_) so the sampling volume
    // can be sanity-checked in RViz.
    // ------------------------------------------------------------
    void publish_dome_marker()
    {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = base_frame_;
        marker.header.stamp = this->now();
        marker.ns = "dome";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.005; // line width
        marker.color.r = 0.1;
        marker.color.g = 0.6;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        constexpr int kNumPoints = 64;
        for (int i = 0; i <= kNumPoints; ++i)
        {
            double angle = 2.0 * M_PI * static_cast<double>(i) / kNumPoints;
            geometry_msgs::msg::Point p;
            p.x = center_x_ + radius_ * std::cos(angle);
            p.y = center_y_ + radius_ * std::sin(angle);
            p.z = center_z_;
            marker.points.push_back(p);
        }
        marker_array.markers.push_back(marker);

        visualization_msgs::msg::Marker sphere;
        sphere.header.frame_id = base_frame_;
        sphere.header.stamp = this->now();
        sphere.ns = "dome";
        sphere.id = 1;
        sphere.type = visualization_msgs::msg::Marker::SPHERE;
        sphere.action = visualization_msgs::msg::Marker::ADD;
        sphere.pose.orientation.w = 1.0;
        sphere.pose.position.x = center_x_;
        sphere.pose.position.y = center_y_;
        sphere.pose.position.z = center_z_;
        sphere.scale.x = radius_ * 2.0;
        sphere.scale.y = radius_ * 2.0;
        sphere.scale.z = radius_ * 2.0;
        sphere.color.r = 1.0;
        sphere.color.g = 0.0;
        sphere.color.b = 0.0;
        sphere.color.a = 0.1;
        marker_array.markers.push_back(sphere);

        marker_pub_->publish(marker_array);
    }

    // ------------------------------------------------------------
    // Attached camera collision object
    // ------------------------------------------------------------
    void attach_camera_object()
    {
        auto xyz = this->get_parameter("camera_mount_xyz").as_double_array();
        auto qxyzw = this->get_parameter("camera_mount_qxyzw").as_double_array();

        if (xyz.size() != 3 || qxyzw.size() != 4)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "attach_camera is true but camera_mount_xyz/camera_mount_qxyzw are not "
                         "set (need 3 and 4 values respectively) -- skipping camera attachment. "
                         "Set them in dome_mover_params.yaml.");
            return;
        }

        auto box_size = this->get_parameter("camera_box_size").as_double_array();
        std::string camera_link_name = this->get_parameter("camera_link_name").as_string();

        shape_msgs::msg::SolidPrimitive box;
        box.type = shape_msgs::msg::SolidPrimitive::BOX;
        box.dimensions = {box_size.at(0), box_size.at(1), box_size.at(2)};

        geometry_msgs::msg::Pose pose;
        pose.position.x = xyz[0];
        pose.position.y = xyz[1];
        pose.position.z = xyz[2];
        pose.orientation.x = qxyzw[0];
        pose.orientation.y = qxyzw[1];
        pose.orientation.z = qxyzw[2];
        pose.orientation.w = qxyzw[3];

        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.header.frame_id = camera_link_name;
        collision_object.id = "camera";
        collision_object.primitives.push_back(box);
        collision_object.primitive_poses.push_back(pose);
        collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;

        moveit_msgs::msg::AttachedCollisionObject attached_object;
        attached_object.link_name = camera_link_name;
        attached_object.object = collision_object;
        attached_object.touch_links = {camera_link_name};

        moveit_msgs::msg::PlanningScene scene;
        scene.is_diff = true;
        scene.robot_state.is_diff = true;
        scene.robot_state.attached_collision_objects.push_back(attached_object);

        if (!scene_client_->wait_for_service(5s))
        {
            RCLCPP_ERROR(this->get_logger(),
                         "/apply_planning_scene service not available -- is move_group running? "
                         "Camera was NOT attached to the planning scene.");
            return;
        }

        auto request = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
        request->scene = scene;

        auto future = scene_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(client_node_, future) != rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_ERROR(this->get_logger(), "/apply_planning_scene call failed.");
            return;
        }

        auto response = future.get();
        if (response->success)
        {
            RCLCPP_INFO(this->get_logger(), "Camera attached to '%s' in the planning scene.",
                        camera_link_name.c_str());
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "/apply_planning_scene reported failure.");
        }
    }

    // ------------------------------------------------------------
    // Dome pose sampling (ported from dome_sampler_pkg/src/dome_sampler.cpp)
    // ------------------------------------------------------------
    tf2::Quaternion look_at_origin(double phi, double theta, double roll)
    {
        tf2::Quaternion q_z, q_y, q_roll, q;
        q_z.setRotation(tf2::Vector3(0, 0, 1), theta);
        q_y.setRotation(tf2::Vector3(0, 1, 0), phi + M_PI / 2.0);
        q_roll.setRotation(tf2::Vector3(1, 0, 0), roll);
        q = q_z * q_y * q_roll;
        return q;
    }

    geometry_msgs::msg::Pose sample_dome_pose()
    {
        std::uniform_real_distribution<double> theta(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> phi(0.0, 1.0);
        std::uniform_real_distribution<double> roll(-M_PI, M_PI);

        double cos_phi_rand = phi(gen_);
        double theta_rand = theta(gen_);
        double roll_rand = roll(gen_);

        double phi_rand = std::acos(cos_phi_rand); // uniform over the hemisphere surface

        double x = radius_ * std::sin(phi_rand) * std::cos(theta_rand);
        double y = radius_ * std::sin(phi_rand) * std::sin(theta_rand);
        double z = radius_ * std::cos(phi_rand);

        geometry_msgs::msg::Pose pose;
        pose.position.x = center_x_ + x;
        pose.position.y = center_y_ + y;
        pose.position.z = center_z_ + z;
        pose.orientation = tf2::toMsg(look_at_origin(phi_rand, theta_rand, roll_rand));
        return pose;
    }

    // ------------------------------------------------------------
    // IK / cartesian-path checks
    // ------------------------------------------------------------
    bool check_ik(const geometry_msgs::msg::Pose &pose)
    {
        if (!ik_client_->wait_for_service(2s))
        {
            RCLCPP_WARN(this->get_logger(), "/compute_ik service not available -- is move_group running?");
            return false;
        }

        auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
        request->ik_request.group_name = planning_group_;
        request->ik_request.pose_stamped.header.frame_id = base_frame_;
        request->ik_request.pose_stamped.pose = pose;
        request->ik_request.avoid_collisions = true;
        request->ik_request.robot_state.is_diff = true;
        request->ik_request.ik_link_name = end_effector_link_;

        auto future = ik_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(client_node_, future) != rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_WARN(this->get_logger(), "IK service call failed");
            return false;
        }

        auto response = future.get();
        return response->error_code.val == response->error_code.SUCCESS;
    }

    bool check_cartesian_path(const geometry_msgs::msg::Pose &pose,
                              trajectory_msgs::msg::JointTrajectory &out_trajectory)
    {
        auto cart_request = std::make_shared<moveit_msgs::srv::GetCartesianPath::Request>();
        cart_request->header.frame_id = base_frame_;
        cart_request->group_name = planning_group_;
        cart_request->waypoints.push_back(pose); // single target -- straight line from current pose
        cart_request->max_step = 0.1;            // 1cm resolution along the path
        cart_request->jump_threshold = 0.0;
        cart_request->avoid_collisions = true;
        cart_request->max_cartesian_speed = 0.1; // 10cm/s max speed for the end effector
        cart_request->link_name = end_effector_link_;

        auto future = plan_client_->async_send_request(cart_request);
        if (rclcpp::spin_until_future_complete(client_node_, future) != rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_WARN(this->get_logger(), "Cartesian path service call failed");
            return false;
        }

        auto response = future.get();
        if (response->error_code.val != response->error_code.SUCCESS || response->fraction < 0.99)
        {
            RCLCPP_WARN(this->get_logger(), "Cartesian path planning failed with error code %d",
                        response->error_code.val);
            return false;
        }

        out_trajectory = response->solution.joint_trajectory;
        return true;
    }

    // ------------------------------------------------------------
    // Filter the planned trajectory down to joints_to_keep_, retime, publish,
    // and block until the motion should have completed.
    // Returns the filtered trajectory (empty if a required joint was missing).
    // ------------------------------------------------------------
    trajectory_msgs::msg::JointTrajectory publish_filtered_trajectory(
        const trajectory_msgs::msg::JointTrajectory &full_traj)
    {
        trajectory_msgs::msg::JointTrajectory filtered;

        std::vector<size_t> keep_indices;
        for (const auto &joint_name : joints_to_keep_)
        {
            auto it = std::find(full_traj.joint_names.begin(), full_traj.joint_names.end(), joint_name);
            if (it == full_traj.joint_names.end())
            {
                RCLCPP_WARN(this->get_logger(),
                            "joints_to_keep entry '%s' not found in planned trajectory -- skipping",
                            joint_name.c_str());
                return trajectory_msgs::msg::JointTrajectory();
            }
            keep_indices.push_back(std::distance(full_traj.joint_names.begin(), it));
            filtered.joint_names.push_back(joint_name);
        }

        for (const auto &pt : full_traj.points)
        {
            trajectory_msgs::msg::JointTrajectoryPoint filtered_pt;
            for (size_t idx : keep_indices)
            {
                filtered_pt.positions.push_back(pt.positions[idx]);
                if (!pt.velocities.empty())
                    filtered_pt.velocities.push_back(pt.velocities[idx]);
                if (!pt.accelerations.empty())
                    filtered_pt.accelerations.push_back(pt.accelerations[idx]);
            }
            filtered_pt.time_from_start = pt.time_from_start;
            filtered.points.push_back(filtered_pt);
        }

        double total_duration_sec = 1.0; // tune this to taste
        size_t n = filtered.points.size();
        for (size_t i = 0; i < n; ++i)
        {
            double t = total_duration_sec * (static_cast<double>(i + 1) / n);
            int32_t sec = static_cast<int32_t>(t);
            uint32_t nanosec = static_cast<uint32_t>((t - sec) * 1e9);
            filtered.points[i].time_from_start = rclcpp::Duration(sec, nanosec);
        }

        traj_pub_->publish(filtered);

        double wait_sec = total_duration_sec + 0.5;
        RCLCPP_INFO(this->get_logger(),
                    "Published planned trajectory (%zu points, %zu joints), waiting %.2fs",
                    filtered.points.size(), filtered.joint_names.size(), wait_sec);

        std::this_thread::sleep_for(std::chrono::duration<double>(wait_sec));
        return filtered;
    }

    // ------------------------------------------------------------
    void handle_move_to_next_point(
        const std::shared_ptr<MoveToNextPoint::Request>,
        std::shared_ptr<MoveToNextPoint::Response> response)
    {
        for (int attempt = 1; attempt <= max_attempts_; ++attempt)
        {
            geometry_msgs::msg::Pose pose = sample_dome_pose();

            // Apply an extra rotation around the Y-axis before IK/planning.
            // BC Z axis is looking in direction of end effector.
            tf2::Quaternion q_orig, q_extra;
            tf2::fromMsg(pose.orientation, q_orig);
            q_extra.setRotation(tf2::Vector3(0, 1, 0), M_PI / 2.0);
            tf2::Quaternion q_new = q_orig * q_extra;
            pose.orientation = tf2::toMsg(q_new);

            if (!check_ik(pose))
            {
                continue;
            }

            trajectory_msgs::msg::JointTrajectory full_traj;
            if (!check_cartesian_path(pose, full_traj))
            {
                continue;
            }

            trajectory_msgs::msg::JointTrajectory filtered = publish_filtered_trajectory(full_traj);
            if (filtered.points.empty())
            {
                continue;
            }

            response->success = true;
            response->message = "Moved to reachable dome pose (attempt " + std::to_string(attempt) +
                                "/" + std::to_string(max_attempts_) + ")";
            response->joint_names = filtered.joint_names;
            response->joint_positions = filtered.points.back().positions;

            RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
            return;
        }

        response->success = false;
        response->message = "No reachable dome pose found after " + std::to_string(max_attempts_) + " attempts.";
        response->joint_names = {};
        response->joint_positions = {};
        RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
    }

    rclcpp::Node::SharedPtr client_node_;
    rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
    rclcpp::Client<moveit_msgs::srv::GetCartesianPath>::SharedPtr plan_client_;
    rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr scene_client_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
    rclcpp::Service<MoveToNextPoint>::SharedPtr service_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr marker_timer_;

    double radius_;
    double center_x_;
    double center_y_;
    double center_z_;
    int max_attempts_;

    std::string planning_group_;
    std::string base_frame_;
    std::vector<std::string> joints_to_keep_;
    std::string end_effector_link_;

    std::random_device rd_;
    std::mt19937 gen_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DomeMover>());
    rclcpp::shutdown();
    return 0;
}
