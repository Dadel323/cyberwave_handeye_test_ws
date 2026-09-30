import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('dome_moveit_pkg')
    default_params = os.path.join(pkg_share, 'config', 'dome_mover_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to dome_mover_params.yaml')
    params_file = LaunchConfiguration('params_file')

    dome_mover = Node(
        package='dome_moveit_pkg',
        executable='dome_mover',
        name='dome_mover',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        params_file_arg,
        dome_mover,
    ])
