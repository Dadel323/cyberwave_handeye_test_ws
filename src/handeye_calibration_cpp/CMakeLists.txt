cmake_minimum_required(VERSION 3.8)
project(handeye_calibration_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(handeye_calibration_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# cv::aruco::CharucoDetector and CharucoBoard::matchImagePoints -- both central
# to the detection code here -- landed in OpenCV 4.7. Earlier versions have only
# the contrib aruco module, whose CharucoBoard has no setLegacyPattern(), so a
# legacy_pattern board cannot be reproduced there at all. Failing loudly at
# configure time beats a wall of template errors, or worse, a build that
# succeeds against a board model that silently differs from the Python one.
if(OpenCV_VERSION VERSION_LESS "4.7.0")
  message(FATAL_ERROR
    "handeye_calibration_cpp requires OpenCV >= 4.7 (found ${OpenCV_VERSION} at "
    "${OpenCV_DIR}).\n"
    "  cv::aruco::CharucoDetector and CharucoBoard::matchImagePoints are 4.7+ APIs, "
    "and setLegacyPattern() -- needed to match the board used by the Python "
    "handeye_calibration package -- does not exist before then.\n"
    "  Install a newer OpenCV and point CMake at it, e.g.:\n"
    "    colcon build --packages-select handeye_calibration_cpp "
    "--cmake-args -DOpenCV_DIR=/path/to/opencv/build")
endif()

set(dependencies
  rclcpp
  sensor_msgs
  std_srvs
  geometry_msgs
  trajectory_msgs
  tf2
  tf2_ros
  cv_bridge
  handeye_calibration_interfaces
)

# ---------------------------------------------------------------------------
# Shared library: board detection, hand-eye solve, dataset I/O.
# ---------------------------------------------------------------------------
add_library(${PROJECT_NAME}_core SHARED
  src/board_utils.cpp
  src/solver.cpp
  src/dataset.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_core PUBLIC ${OpenCV_LIBS} yaml-cpp)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
foreach(node
    capture_node
    calibration_node
    auto_calibration_node
    intrinsics_node
    run_points_node
    viewer_node)
  add_executable(${node} src/${node}.cpp)
  target_link_libraries(${node} ${PROJECT_NAME}_core)
  ament_target_dependencies(${node} ${dependencies})
  install(TARGETS ${node} DESTINATION lib/${PROJECT_NAME})
endforeach()

install(TARGETS ${PROJECT_NAME}_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core)
ament_export_dependencies(${dependencies} OpenCV yaml-cpp)
ament_package()
