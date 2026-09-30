import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('handeye_calibration_cpp')
    default_params = os.path.join(pkg_share, 'config', 'pipeline_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to pipeline_params.yaml')

    params_file = LaunchConfiguration('params_file')

    calibration_node = Node(
        package='handeye_calibration_cpp',
        executable='calibration_node',
        name='calibration_node',
        output='screen',
        parameters=[params_file],
    )

    auto_calibration_node = Node(
        package='handeye_calibration_cpp',
        executable='auto_calibration_node',
        name='auto_calibration_node',
        output='screen',
        parameters=[params_file],
    )
    
    run_points_node = Node(
        package='handeye_calibration_cpp',
        executable='run_points_node',
        name='run_points_node',
        output='screen',
        parameters=[params_file]
    )

    capture_node = Node(
        package='handeye_calibration_cpp',
        executable='capture_node',
        name='capture_node',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        params_file_arg,
        calibration_node,
        auto_calibration_node,
        run_points_node,
        capture_node,
    ])
