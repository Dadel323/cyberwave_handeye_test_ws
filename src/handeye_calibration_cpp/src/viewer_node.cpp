// viewer_node.cpp
//
// Continuous live view of the calibration camera, taken from the compressed
// image transport (`<image_topic>/compressed`, sensor_msgs/CompressedImage).
//
// Subscribing to the compressed stream rather than image_raw keeps a viewer off
// the hot path: image_raw at 1280x720 bgr8 is ~2.8 MB per frame, and every extra
// subscriber costs another copy of that through the middleware. The compressed
// topic is roughly a tenth of it, and JPEG artifacts in a monitoring window do
// not matter -- nothing here feeds a solve. (The detection nodes deliberately
// stay on image_raw for that reason.)
//
// The window needs a display, so run it from a session that has one (not over a
// plain ssh without X forwarding). Press q or Esc in the window to quit.
//
//     ros2 run handeye_calibration_cpp viewer_node
//     ros2 run handeye_calibration_cpp viewer_node --ros-args -p
//     image_topic:=/camera/color/image_raw

#include <memory>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

using sensor_msgs::msg::CompressedImage;

class ViewerNode : public rclcpp::Node
{
public:
  ViewerNode()
  : Node("viewer_node")
  {
    // The base topic, matching the `image_topic` parameter every other node in
    // this package takes; '/compressed' is appended here so switching cameras
    // means changing the same value in both places.
    declare_parameter("image_topic", "/webcam/image_raw");
    declare_parameter("window_name", "camera");
    declare_parameter("show_fps", true);

    std::string base_topic = get_parameter("image_topic").as_string();
    while (!base_topic.empty() && base_topic.back() == '/') {
      base_topic.pop_back();
    }
    topic_ = base_topic + "/compressed";
    window_name_ = get_parameter("window_name").as_string();
    show_fps_ = get_parameter("show_fps").as_bool();

    // Sensor-data QoS (best effort, depth 5): a viewer that falls behind
    // should drop frames rather than queue stale ones, and image_transport
    // publishers offer best-effort -- a RELIABLE subscription would simply
    // never match.
    sub_ = create_subscription<CompressedImage>(
      topic_, rclcpp::SensorDataQoS(),
      std::bind(&ViewerNode::image_cb, this, std::placeholders::_1));

    // Drives the GUI event loop and the quit key. cv::imshow is called from
    // here rather than from the image callback so the window stays responsive
    // even when no frames are arriving.
    fps_last_stamp_ = now();
    gui_timer_ = create_wall_timer(
      std::chrono::milliseconds(30), std::bind(&ViewerNode::gui_tick, this));
    fps_timer_ = create_wall_timer(
      std::chrono::seconds(1), std::bind(&ViewerNode::fps_tick, this));

    RCLCPP_INFO(
      get_logger(), "viewer_node showing %s. Press q or Esc in the window to quit.",
      topic_.c_str());
  }

private:
  void image_cb(const CompressedImage::SharedPtr msg)
  {
    const cv::Mat buf(1, static_cast<int>(msg->data.size()), CV_8U, msg->data.data());
    const cv::Mat bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      RCLCPP_WARN(
        get_logger(), "Could not decode a \"%s\" frame -- skipping it.",
        msg->format.c_str());
      return;
    }
    latest_bgr_ = bgr;
    ++frame_count_;
  }

  /// Measured over wall-clock so a stalled stream reads as 0, not as the last
  /// good rate.
  void fps_tick()
  {
    const rclcpp::Time current = now();
    const double elapsed = (current - fps_last_stamp_).seconds();
    if (elapsed > 0.0) {
      fps_ = static_cast<double>(frame_count_ - fps_last_count_) / elapsed;
    }
    fps_last_count_ = frame_count_;
    fps_last_stamp_ = current;

    if (frame_count_ == 0 && !warned_no_frames_) {
      RCLCPP_WARN(
        get_logger(),
        "No frames on %s yet. Is the camera up, and is compressed_image_transport "
        "installed alongside it?",
        topic_.c_str());
      warned_no_frames_ = true;
    }
  }

  void gui_tick()
  {
    if (latest_bgr_.empty()) {
      return;
    }

    cv::Mat frame = latest_bgr_;
    if (show_fps_) {
      frame = frame.clone();
      char label[64];
      std::snprintf(
        label, sizeof(label), "%4.1f fps  %dx%d", fps_, frame.cols, frame.rows);
      cv::putText(
        frame, label, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    cv::imshow(window_name_, frame);
    const int key = cv::waitKey(1) & 0xFF;
    if (key == 'q' || key == 27) {
      RCLCPP_INFO(get_logger(), "Quit key pressed -- shutting down.");
      rclcpp::shutdown();
    }
  }

  std::string topic_;
  std::string window_name_;
  bool show_fps_ = true;

  cv::Mat latest_bgr_;
  uint64_t frame_count_ = 0;
  uint64_t fps_last_count_ = 0;
  double fps_ = 0.0;
  rclcpp::Time fps_last_stamp_;
  bool warned_no_frames_ = false;

  rclcpp::Subscription<CompressedImage>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr gui_timer_;
  rclcpp::TimerBase::SharedPtr fps_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ViewerNode>());
  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}
