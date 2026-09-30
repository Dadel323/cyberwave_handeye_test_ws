"""
manual.launch.py

The half of the hand-eye pipeline that runs unattended, for the workflow where
you place the arm by hand: capture_node (which samples image + TF on request)
and calibration_node (which solves once the dataset is complete).

auto_calibration_node is deliberately NOT started here. In manual mode it
prompts with input(), and a process started by ros2 launch has no terminal on
stdin -- the prompt would never return and the run would hang with no error.
Run it yourself, in its own terminal:

  ros2 launch handeye_calibration manual.launch.py \
      params_file:=<pkg>/config/so100_params.yaml

  # second terminal
  ros2 run handeye_calibration auto_calibration_node --ros-args \
      --params-file <pkg>/config/so100_params.yaml

The arm (motor_bridge + robot_state_publisher) and the camera
(webcam.launch.py) come up separately, as they do for the replay pipeline.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('handeye_calibration')
    default_params = os.path.join(pkg_share, 'config', 'so100_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Pipeline parameters (board, frames, output_dir).')
    params_file = LaunchConfiguration('params_file')

    capture_node = Node(
        package='handeye_calibration',
        executable='capture_node',
        name='capture_node',
        output='screen',
        parameters=[params_file],
    )

    calibration_node = Node(
        package='handeye_calibration',
        executable='calibration_node',
        name='calibration_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([params_file_arg, capture_node, calibration_node])
