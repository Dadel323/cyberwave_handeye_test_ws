// intrinsics_node.cpp
//
// Monocular intrinsics calibration for a plain webcam, using the same ChArUco
// board the hand-eye pipeline already uses.
//
// A webcam publishes an all-zero CameraInfo (unlike a depth camera, which ships
// factory intrinsics), so capture_node has nothing usable to write into
// dataset.yaml. This node produces those intrinsics.
//
// The board is built via board_utils::build_board -- the same function
// capture_node/calibration_node use -- so the board model (including
// legacy_pattern) is identical across intrinsics and the hand-eye solve.
//
// Services:
//   ~/capture    (std_srvs/Trigger) -- detect the board in the current frame and
//                                      store the correspondences as one view
//   ~/calibrate  (std_srvs/Trigger) -- solve over all stored views and write the
//                                      camera_info YAML
//   ~/reset      (std_srvs/Trigger) -- drop all stored views

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "handeye_calibration_cpp/board_utils.hpp"

namespace fs = std::filesystem;
using std_srvs::srv::Trigger;
using namespace handeye_calibration_cpp;

class IntrinsicsNode : public rclcpp::Node
{
public:
  IntrinsicsNode()
  : Node("intrinsics_node")
  {
    declare_parameter("image_topic", "/webcam/image_raw");
    declare_parameter("output_path", "");
    declare_parameter("camera_name", "webcam");
    declare_parameter("min_views", 10);
    declare_parameter("debug_dir", "");

    // -- same board parameters as capture_node --
    declare_parameter("squares_x", 8);
    declare_parameter("squares_y", 10);
    declare_parameter("square_length_m", 0.015);
    declare_parameter("marker_length_m", 0.011);
    declare_parameter("dictionary", "4X4_50");
    declare_parameter("legacy_pattern", false);
    declare_parameter("min_charuco_corners", 6);

    target_ = build_board(
      static_cast<int>(get_parameter("squares_x").as_int()),
      static_cast<int>(get_parameter("squares_y").as_int()),
      get_parameter("square_length_m").as_double(),
      get_parameter("marker_length_m").as_double(),
      get_parameter("dictionary").as_string(),
      get_parameter("legacy_pattern").as_bool());
    min_corners_ = static_cast<int>(get_parameter("min_charuco_corners").as_int());
    min_views_ = static_cast<int>(get_parameter("min_views").as_int());

    const std::string output_path = get_parameter("output_path").as_string();
    if (output_path.empty()) {
      RCLCPP_ERROR(
        get_logger(), "Parameter \"output_path\" is unset -- nowhere to write the result.");
      throw std::runtime_error("output_path is unset");
    }
    output_path_ = output_path;

    const std::string debug_dir = get_parameter("debug_dir").as_string();
    if (!debug_dir.empty()) {
      debug_dir_ = debug_dir;
      fs::create_directories(debug_dir_);
      RCLCPP_INFO(get_logger(), "Saving annotated views to %s", debug_dir_.c_str());
    }

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      get_parameter("image_topic").as_string(), 10,
      std::bind(&IntrinsicsNode::image_cb, this, std::placeholders::_1));

    capture_srv_ = create_service<Trigger>(
      "~/capture",
      std::bind(
        &IntrinsicsNode::srv_capture, this, std::placeholders::_1, std::placeholders::_2));
    calibrate_srv_ = create_service<Trigger>(
      "~/calibrate",
      std::bind(
        &IntrinsicsNode::srv_calibrate, this, std::placeholders::_1,
        std::placeholders::_2));
    reset_srv_ = create_service<Trigger>(
      "~/reset",
      std::bind(
        &IntrinsicsNode::srv_reset, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(
      get_logger(), "intrinsics_node ready. Services: ~/capture, ~/calibrate, ~/reset");
  }

private:
  void image_cb(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      latest_bgr_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN(get_logger(), "cv_bridge conversion failed: %s", e.what());
    }
  }

  // ------------------------------------------------------------
  void srv_capture(
    const std::shared_ptr<Trigger::Request> /*request*/,
    std::shared_ptr<Trigger::Response> response)
  {
    if (latest_bgr_.empty()) {
      response->success = false;
      response->message = "No camera frame received yet.";
      RCLCPP_WARN(get_logger(), "%s", response->message.c_str());
      return;
    }

    const cv::Mat bgr = latest_bgr_.clone();
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat charuco_corners, charuco_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<int> marker_ids;
    target_->detector.detectBoard(
      gray, charuco_corners, charuco_ids, marker_corners, marker_ids);

    const int found = charuco_ids.empty() ? 0 : static_cast<int>(charuco_ids.total());
    if (found < min_corners_) {
      response->success = false;
      response->message = "Board not detected (" + std::to_string(found) +
        " corners, need " + std::to_string(min_corners_) + ").";
      RCLCPP_WARN(get_logger(), "%s", response->message.c_str());
      return;
    }

    cv::Mat obj_pts, img_pts;
    target_->board.matchImagePoints(charuco_corners, charuco_ids, obj_pts, img_pts);
    obj_points_.push_back(obj_pts);
    img_points_.push_back(img_pts);
    image_size_ = cv::Size(gray.cols, gray.rows);

    const size_t n = obj_points_.size();
    if (!debug_dir_.empty()) {
      frames_.push_back(bgr);
      cv::Mat annotated = bgr.clone();
      if (!marker_ids.empty()) {
        cv::aruco::drawDetectedMarkers(annotated, marker_corners, marker_ids);
      }
      cv::aruco::drawDetectedCornersCharuco(
        annotated, charuco_corners, charuco_ids, cv::Scalar(0, 0, 255));
      cv::imwrite((fs::path(debug_dir_) / view_name("view_%03zu.png", n)).string(), annotated);
    }

    response->success = true;
    response->message = "View " + std::to_string(n) + " captured (" +
      std::to_string(found) + " corners).";
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
  }

  // ------------------------------------------------------------
  void srv_calibrate(
    const std::shared_ptr<Trigger::Request> /*request*/,
    std::shared_ptr<Trigger::Response> response)
  {
    const size_t n = obj_points_.size();
    if (n < 3) {
      response->success = false;
      response->message = "Only " + std::to_string(n) + " views -- need at least 3.";
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }
    if (n < static_cast<size_t>(min_views_)) {
      RCLCPP_WARN(
        get_logger(),
        "Only %zu views (recommended >= %d) -- distortion terms will be poorly "
        "constrained.",
        n, min_views_);
    }

    cv::Mat K, dist;
    std::vector<cv::Mat> rvecs, tvecs;
    const double rms =
      cv::calibrateCamera(obj_points_, img_points_, image_size_, K, dist, rvecs, tvecs);

    const double total_error = report_per_view(K, dist, rvecs, tvecs);
    write_undistorted(K, dist);
    write_camera_info(K, dist);

    std::ostringstream msg;
    msg << "Calibrated over " << n << " views, total error " << std::fixed
        << std::setprecision(4) << total_error << ". Wrote " << output_path_;
    response->success = true;
    response->message = msg.str();
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    RCLCPP_INFO(
      get_logger(), "  fx=%.2f fy=%.2f cx=%.2f cy=%.2f", K.at<double>(0, 0),
      K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2));
    RCLCPP_INFO(get_logger(), "  calibrateCamera rms: %.4f px", rms);
    if (rms > 1.0) {
      RCLCPP_WARN(
        get_logger(),
        "calibrateCamera rms > 1.0 px. The per-view table above is sorted worst "
        "first -- a few views far above the rest means drop those and recapture; "
        "every view alike means a systematic cause (board model, a focus change "
        "mid-run, or edge overshoot from high sharpness).");
    }
  }

  /// Reprojection error per view, worst first; returns the mean over views.
  ///
  /// The error is the one the OpenCV calibration tutorial defines: the L2 norm
  /// of (detected - reprojected) over a view's points, divided by the point
  /// count, then averaged across views.
  ///
  /// The divisor is N rather than sqrt(N), so this is not an RMS in pixels and
  /// does not match the rms calibrateCamera returns -- it is smaller by roughly
  /// sqrt(N), and for the same actual error a view with more corners scores
  /// lower than one with fewer. Compare views here only when their corner
  /// counts are close; calibrateCamera's rms, logged separately, is the figure
  /// to judge in pixels.
  ///
  /// The total alone hides one bad view among the good ones, which is why the
  /// per-view breakdown is printed: a couple of blurred captures and a
  /// systematic error affecting every view have completely different fixes.
  double report_per_view(
    const cv::Mat & K, const cv::Mat & dist, const std::vector<cv::Mat> & rvecs,
    const std::vector<cv::Mat> & tvecs)
  {
    struct ViewError
    {
      double error;
      size_t view;             ///< 1-based, as printed
      int num_points;
      std::vector<cv::Point2f> projected;
      std::vector<cv::Point2f> detected;
    };

    std::vector<ViewError> errors;
    double mean_error = 0.0;
    for (size_t i = 0; i < obj_points_.size(); ++i) {
      cv::Mat projected;
      cv::projectPoints(obj_points_[i], rvecs[i], tvecs[i], K, dist, projected);
      // projectPoints returns float64 while matchImagePoints gives float32, and
      // cv::norm rejects a mixed pair, so the projection is cast back.
      projected.convertTo(projected, img_points_[i].type());

      const double error =
        cv::norm(img_points_[i], projected, cv::NORM_L2) / projected.total();
      mean_error += error;

      ViewError ve;
      ve.error = error;
      ve.view = i + 1;
      ve.num_points = static_cast<int>(img_points_[i].total());
      projected.reshape(2, static_cast<int>(projected.total())).copyTo(ve.projected);
      img_points_[i].reshape(2, static_cast<int>(img_points_[i].total()))
      .copyTo(ve.detected);
      errors.push_back(std::move(ve));
    }
    const double total_error = mean_error / static_cast<double>(obj_points_.size());

    std::vector<const ViewError *> sorted;
    sorted.reserve(errors.size());
    for (const auto & e : errors) {
      sorted.push_back(&e);
    }
    std::sort(
      sorted.begin(), sorted.end(),
      [](const ViewError * a, const ViewError * b) {return a->error > b->error;});

    RCLCPP_INFO(get_logger(), "  per-view reprojection error (worst first):");
    for (const auto * e : sorted) {
      RCLCPP_INFO(
        get_logger(), "    view %3zu  %3d corners  %8.5f", e->view, e->num_points,
        e->error);
    }
    RCLCPP_INFO(get_logger(), "  total error: %g", total_error);

    if (!debug_dir_.empty()) {
      for (const auto & e : errors) {
        cv::Mat annotated = frames_[e.view - 1].clone();
        for (size_t p = 0; p < e.detected.size() && p < e.projected.size(); ++p) {
          cv::circle(
            annotated, cv::Point(cvRound(e.detected[p].x), cvRound(e.detected[p].y)), 4,
            cv::Scalar(0, 255, 0), 1);                              // detected
          cv::circle(
            annotated, cv::Point(cvRound(e.projected[p].x), cvRound(e.projected[p].y)), 2,
            cv::Scalar(0, 0, 255), -1);                             // reprojected
        }
        char label[64];
        std::snprintf(label, sizeof(label), "view %zu  error %.5f", e.view, e.error);
        cv::putText(
          annotated, label, cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
          cv::Scalar(0, 0, 255), 2);
        cv::imwrite(
          (fs::path(debug_dir_) / view_name("view_%03zu_reproj.png", e.view)).string(),
          annotated);
      }
      RCLCPP_INFO(get_logger(), "  annotated views written to %s", debug_dir_.c_str());
    }

    return total_error;
  }

  /// Undistort every captured view and re-detect the board on the result.
  ///
  /// Detection is re-run rather than the stored corners reused: if the
  /// distortion model is right, the board's rows and columns come out straight
  /// and every marker still decodes. A view where undistortion bends the board,
  /// or loses markers it had before, is the visible form of an over-fitted
  /// distortion solve -- which the reprojection error will not show, because
  /// that is measured in the distorted frame where the model was fitted.
  void write_undistorted(const cv::Mat & K, const cv::Mat & dist)
  {
    if (debug_dir_.empty()) {
      return;
    }
    for (size_t i = 0; i < frames_.size(); ++i) {
      cv::Mat undistorted;
      cv::undistort(frames_[i], undistorted, K, dist);
      cv::Mat gray;
      cv::cvtColor(undistorted, gray, cv::COLOR_BGR2GRAY);

      cv::Mat charuco_corners, charuco_ids;
      std::vector<std::vector<cv::Point2f>> marker_corners;
      std::vector<int> marker_ids;
      target_->detector.detectBoard(
        gray, charuco_corners, charuco_ids, marker_corners, marker_ids);

      if (!marker_ids.empty()) {
        cv::aruco::drawDetectedMarkers(undistorted, marker_corners, marker_ids);
      }
      int found = 0;
      if (!charuco_ids.empty()) {
        found = static_cast<int>(charuco_ids.total());
        cv::aruco::drawDetectedCornersCharuco(
          undistorted, charuco_corners, charuco_ids, cv::Scalar(0, 0, 255));
      }
      char label[80];
      std::snprintf(
        label, sizeof(label), "view %zu undistorted  %d corners", i + 1, found);
      cv::putText(
        undistorted, label, cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 0, 255), 2);
      cv::imwrite(
        (fs::path(debug_dir_) / view_name("view_%03zu_undistorted.png", i + 1)).string(),
        undistorted);
    }
    RCLCPP_INFO(get_logger(), "  undistorted views written to %s", debug_dir_.c_str());
  }

  /// camera_info_manager YAML layout -- this is what usb_cam's camera_info_url
  /// loads and republishes as CameraInfo.
  void write_camera_info(const cv::Mat & K, const cv::Mat & dist)
  {
    const cv::Mat d = dist.reshape(1, 1);

    std::ostringstream out;
    out << "image_width: " << image_size_.width << "\n"
        << "image_height: " << image_size_.height << "\n"
        << "camera_name: " << get_parameter("camera_name").as_string() << "\n"
        << "camera_matrix:\n  rows: 3\n  cols: 3\n  data: "
        << list_of(
      {K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2)})
        << "\n"
        << "distortion_model: plumb_bob\n"
        << "distortion_coefficients:\n  rows: 1\n  cols: " << d.cols << "\n  data: ";

    std::vector<double> dv;
    for (int i = 0; i < d.cols; ++i) {
      dv.push_back(d.at<double>(0, i));
    }
    out << list_of(dv) << "\n"
        << "rectification_matrix:\n  rows: 3\n  cols: 3\n"
        << "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n"
        << "projection_matrix:\n  rows: 3\n  cols: 4\n  data: "
        << list_of(
      {K.at<double>(0, 0), 0.0, K.at<double>(0, 2), 0.0,
        0.0, K.at<double>(1, 1), K.at<double>(1, 2), 0.0,
        0.0, 0.0, 1.0, 0.0})
        << "\n";

    const fs::path path(output_path_);
    if (path.has_parent_path()) {
      fs::create_directories(path.parent_path());
    }
    std::ofstream file(path);
    if (!file) {
      RCLCPP_ERROR(get_logger(), "Could not open %s for writing.", output_path_.c_str());
      return;
    }
    file << out.str();
  }

  static std::string list_of(const std::vector<double> & values)
  {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << values[i];
    }
    oss << "]";
    return oss.str();
  }

  static std::string view_name(const char * format, size_t n)
  {
    char buf[64];
    std::snprintf(buf, sizeof(buf), format, n);
    return buf;
  }

  // ------------------------------------------------------------
  void srv_reset(
    const std::shared_ptr<Trigger::Request> /*request*/,
    std::shared_ptr<Trigger::Response> response)
  {
    obj_points_.clear();
    img_points_.clear();
    frames_.clear();
    response->success = true;
    response->message = "Cleared all stored views.";
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
  }

  std::shared_ptr<CharucoTarget> target_;
  int min_corners_ = 6;
  int min_views_ = 10;
  std::string output_path_;
  std::string debug_dir_;

  cv::Mat latest_bgr_;
  cv::Size image_size_;
  std::vector<cv::Mat> obj_points_;
  std::vector<cv::Mat> img_points_;
  std::vector<cv::Mat> frames_;    // kept only to annotate reprojections later

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Service<Trigger>::SharedPtr capture_srv_;
  rclcpp::Service<Trigger>::SharedPtr calibrate_srv_;
  rclcpp::Service<Trigger>::SharedPtr reset_srv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<IntrinsicsNode>());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("intrinsics_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
