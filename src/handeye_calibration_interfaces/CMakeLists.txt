cmake_minimum_required(VERSION 3.8)
project(handeye_calibration_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/CalibrateDataset.srv"
  "srv/SamplePosition.srv"
  "srv/MoveToNextPoint.srv"
  "srv/SampleScene.srv"
)

ament_package()
