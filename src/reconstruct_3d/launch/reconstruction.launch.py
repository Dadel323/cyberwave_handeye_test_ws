#!/usr/bin/env python3
import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import xacro


def generate_launch_description():
    # ------------------------------------------------------------
    # Launch arguments -- override any of these from the CLI, e.g.:
    #   ros2 launch <pkg> reconstruction_pipeline.launch.py num_samples:=15
    # ------------------------------------------------------------
    
    pkg_share = get_package_share_directory('reconstruct_3d')
    default_params = os.path.join(pkg_share, 'config', 'config.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to config.yaml')
    params_file = LaunchConfiguration('params_file')

    gripper_to_camera_tf = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='gripper_to_camera_tf',
    arguments=['--x', '0.06924787', '--y', '0.00162293', '--z', '0.06550527',
               '--qx', '-0.06521939', '--qy', '-0.06155832',
               '--qz', '0.72224376', '--qw', '0.6857995',
               '--frame-id', 'openarm_right_hand',
               '--child-frame-id', 'camera_link'], #'camera_link',],
    )


    run_points_node = Node(
        package='handeye_calibration',
        executable='run_points_node',
        name='run_points_node',
        output='screen',
        parameters=[params_file],
    )

    scene_capture_node = Node(
        package='reconstruct_3d',
        executable='scene_capture_node',
        name='scene_capture_node',
        output='screen',
        parameters=[params_file],
    )

    reconstruct_3d_node = Node(
        package='reconstruct_3d',
        executable='reconstruct_3d_node',
        name='reconstruct_3d_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        params_file_arg,
        #gripper_to_camera_tf,
        #run_points_node,
        scene_capture_node,
        reconstruct_3d_node,
    ])