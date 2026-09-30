#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <random>
#include <cmath>
#include <chrono>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class DomeSampler : public rclcpp::Node
{
public:
    DomeSampler()
        : Node("dome_sampler"), gen_(rd_())
    {
        this->declare_parameter("radius", 0.2);
        this->declare_parameter("num_samples", 200);
        this->declare_parameter("center_x", -0.18);
        this->declare_parameter("center_y", 0.34);
        this->declare_parameter("center_z", 0.2);

        center_x_ = this->get_parameter("center_x").as_double();
        center_y_ = this->get_parameter("center_y").as_double();
        center_z_ = this->get_parameter("center_z").as_double();

        radius_ = this->get_parameter("radius").as_double();
        num_samples_ = this->get_parameter("num_samples").as_int();

        pose_array_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            "/dome_samples", 10);

        timer_ = this->create_wall_timer(
            std::chrono::seconds(20), std::bind(&DomeSampler::publish_samples, this));
    }

private:
    tf2::Quaternion look_at_origin(double phi, double theta, double roll)
    {
        tf2::Quaternion q_z, q_y, q_roll, q;
        q_z.setRotation(tf2::Vector3(0, 0, 1), theta);
        q_y.setRotation(tf2::Vector3(0, 1, 0), phi + M_PI / 2.0);
        q_roll.setRotation(tf2::Vector3(1, 0, 0), roll);
        q = q_z * q_y * q_roll;

        return q;
    }
    tf2::Quaternion look_at_origin_z(double phi, double theta, double roll)
    {
        tf2::Quaternion q_z, q_y, q_roll, q;
        q_z.setRotation(tf2::Vector3(0, 0, 1), theta);
        q_y.setRotation(tf2::Vector3(0, 1, 0), phi + M_PI); // was phi + M_PI/2 for X-forward
        q_roll.setRotation(tf2::Vector3(0, 0, 1), roll);    // was (1,0,0) for X-forward

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

        geometry_msgs::msg::Pose target_point;
        target_point.position.x = center_x_ + x;
        target_point.position.y = center_y_ + y;
        target_point.position.z = center_z_ + z;
        target_point.orientation = tf2::toMsg(look_at_origin(phi_rand, theta_rand, roll_rand));

        return target_point;
    }
    void publish_samples()
    {
        geometry_msgs::msg::PoseArray pose_array;
        pose_array.header.frame_id = "world";
        pose_array.header.stamp = this->now();

        for (int i = 0; i < num_samples_; ++i)
        {
            pose_array.poses.push_back(
                sample_dome_pose());
        }

        pose_array_pub_->publish(pose_array);

        RCLCPP_INFO(this->get_logger(), "Published %d dome sample poses on /dome_samples",
                    num_samples_);
    }

    double radius_;
    int num_samples_;
    double center_x_;
    double center_y_;
    double center_z_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::random_device rd_;
    std::mt19937 gen_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DomeSampler>());
    rclcpp::shutdown();
    return 0;
}