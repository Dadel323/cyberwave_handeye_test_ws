#include "handeye_calibration_cpp/dataset.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace handeye_calibration_cpp
{

namespace
{
/// A 3x3 rotation stored as a nested list of lists.
cv::Matx33d parse_matrix3(const YAML::Node & node, const std::string & what)
{
  if (!node || !node.IsSequence() || node.size() != 3) {
    throw std::runtime_error(what + " is not a 3x3 matrix.");
  }
  cv::Matx33d m;
  for (int r = 0; r < 3; ++r) {
    const YAML::Node row = node[r];
    if (!row.IsSequence() || row.size() != 3) {
      throw std::runtime_error(what + " is not a 3x3 matrix.");
    }
    for (int c = 0; c < 3; ++c) {
      m(r, c) = row[c].as<double>();
    }
  }
  return m;
}

cv::Vec3d parse_vec3(const YAML::Node & node, const std::string & what)
{
  if (!node || !node.IsSequence() || node.size() != 3) {
    throw std::runtime_error(what + " is not a 3-vector.");
  }
  return cv::Vec3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

YAML::Node emit_matrix3(const cv::Matx33d & m)
{
  YAML::Node node(YAML::NodeType::Sequence);
  for (int r = 0; r < 3; ++r) {
    YAML::Node row(YAML::NodeType::Sequence);
    for (int c = 0; c < 3; ++c) {
      row.push_back(m(r, c));
    }
    node.push_back(row);
  }
  return node;
}
}  // namespace

Dataset load_dataset(const std::string & dataset_dir)
{
  const fs::path path = fs::path(dataset_dir) / "dataset.yaml";
  if (!fs::exists(path)) {
    throw std::runtime_error("Dataset not found: " + path.string());
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("Could not parse " + path.string() + ": " + e.what());
  }

  Dataset dataset;

  const cv::Matx33d k = parse_matrix3(root["camera_matrix"], "camera_matrix");
  dataset.camera_matrix = cv::Mat(k).clone();

  const YAML::Node dist = root["dist_coeffs"];
  if (!dist || !dist.IsSequence()) {
    throw std::runtime_error("dist_coeffs is missing or not a list.");
  }
  dataset.dist_coeffs = cv::Mat(1, static_cast<int>(dist.size()), CV_64F);
  for (size_t i = 0; i < dist.size(); ++i) {
    dataset.dist_coeffs.at<double>(0, static_cast<int>(i)) = dist[i].as<double>();
  }

  const YAML::Node board = root["board"];
  if (!board) {
    throw std::runtime_error("dataset.yaml has no 'board' block.");
  }
  dataset.board.squares_x = board["squares_x"].as<int>();
  dataset.board.squares_y = board["squares_y"].as<int>();
  dataset.board.square_length_m = board["square_length_m"].as<double>();
  dataset.board.marker_length_m = board["marker_length_m"].as<double>();
  dataset.board.dictionary = board["dictionary"].as<std::string>();
  dataset.board.legacy_pattern = board["legacy_pattern"].as<bool>();
  dataset.board.min_charuco_corners = board["min_charuco_corners"].as<int>();

  const YAML::Node entries = root["entries"];
  if (entries && entries.IsSequence()) {
    for (const auto & entry : entries) {
      DatasetEntry e;
      e.image_file = entry["image_file"].as<std::string>();
      e.R_gripper2base = parse_matrix3(entry["R_gripper2base"], "R_gripper2base");
      e.t_gripper2base = parse_vec3(entry["t_gripper2base"], "t_gripper2base");
      dataset.entries.push_back(e);
    }
  }
  return dataset;
}

void save_dataset(const std::string & dataset_dir, const Dataset & dataset)
{
  YAML::Node root;

  // capture_node fills these from CameraInfo, which is float64; converting
  // explicitly means a dataset written from any other source still comes out
  // right rather than being reinterpreted bit-for-bit.
  cv::Mat km, dm;
  dataset.camera_matrix.reshape(1, 3).convertTo(km, CV_64F);
  dataset.dist_coeffs.reshape(1, 1).convertTo(dm, CV_64F);

  cv::Matx33d k;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      k(r, c) = km.at<double>(r, c);
    }
  }
  root["camera_matrix"] = emit_matrix3(k);

  YAML::Node dist(YAML::NodeType::Sequence);
  for (int i = 0; i < dm.cols; ++i) {
    dist.push_back(dm.at<double>(0, i));
  }
  root["dist_coeffs"] = dist;

  YAML::Node board;
  board["squares_x"] = dataset.board.squares_x;
  board["squares_y"] = dataset.board.squares_y;
  board["square_length_m"] = dataset.board.square_length_m;
  board["marker_length_m"] = dataset.board.marker_length_m;
  board["dictionary"] = dataset.board.dictionary;
  board["legacy_pattern"] = dataset.board.legacy_pattern;
  board["min_charuco_corners"] = dataset.board.min_charuco_corners;
  root["board"] = board;

  YAML::Node entries(YAML::NodeType::Sequence);
  for (const auto & e : dataset.entries) {
    YAML::Node entry;
    entry["image_file"] = e.image_file;
    entry["R_gripper2base"] = emit_matrix3(e.R_gripper2base);
    YAML::Node t(YAML::NodeType::Sequence);
    t.push_back(e.t_gripper2base[0]);
    t.push_back(e.t_gripper2base[1]);
    t.push_back(e.t_gripper2base[2]);
    entry["t_gripper2base"] = t;
    entries.push_back(entry);
  }
  root["entries"] = entries;

  const fs::path path = fs::path(dataset_dir) / "dataset.yaml";
  fs::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open " + path.string() + " for writing.");
  }
  out << root;
  out << "\n";
}

void save_handeye_result(const std::string & path, const HandEyeResultFile & result)
{
  YAML::Node root;
  root["parent_frame"] = result.parent_frame;
  root["child_frame"] = result.child_frame;

  YAML::Node translation;
  translation["x"] = result.translation[0];
  translation["y"] = result.translation[1];
  translation["z"] = result.translation[2];
  root["translation"] = translation;

  YAML::Node rotation;
  rotation["x"] = result.rotation_xyzw[0];
  rotation["y"] = result.rotation_xyzw[1];
  rotation["z"] = result.rotation_xyzw[2];
  rotation["w"] = result.rotation_xyzw[3];
  root["rotation_xyzw"] = rotation;

  root["method"] = result.method;
  root["num_samples"] = result.num_samples;
  root["num_skipped"] = result.num_skipped;

  const fs::path out_path(path);
  if (out_path.has_parent_path()) {
    fs::create_directories(out_path.parent_path());
  }
  std::ofstream out(out_path);
  if (!out) {
    throw std::runtime_error("Could not open " + path + " for writing.");
  }
  out << root;
  out << "\n";
}

std::vector<LoadedSample> load_samples(
  const std::string & dataset_dir, const Dataset & dataset,
  std::vector<SkippedEntry> & skipped)
{
  auto target = build_board(dataset.board);

  std::vector<LoadedSample> samples;
  for (size_t index = 0; index < dataset.entries.size(); ++index) {
    const DatasetEntry & entry = dataset.entries[index];
    const fs::path image_path = fs::path(dataset_dir) / "images" / entry.image_file;

    const cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      skipped.push_back({index, entry.image_file, "image unreadable"});
      continue;
    }

    const auto pose = detect_board_pose(
      img, *target, dataset.camera_matrix, dataset.dist_coeffs,
      dataset.board.min_charuco_corners);
    if (!pose) {
      skipped.push_back({index, entry.image_file, "board not detected"});
      continue;
    }

    samples.push_back(
      LoadedSample{
        index, entry.image_file,
        Sample{entry.R_gripper2base, entry.t_gripper2base, pose->R, pose->t}});
  }
  return samples;
}

}  // namespace handeye_calibration_cpp
