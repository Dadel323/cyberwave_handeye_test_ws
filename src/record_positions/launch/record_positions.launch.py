import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('record_positions')
    default_params = os.path.join(pkg_share, 'config', 'record_positions_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to record_positions_params.yaml')
    params_file = LaunchConfiguration('params_file')

    snapshot_node = Node(
        package='record_positions',
        executable='snapshot_node',
        name='snapshot_node',
        output='screen',
        parameters=[params_file],
    )

    record_positions_node = Node(
        package='record_positions',
        executable='record_positions_node',
        name='record_positions_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        params_file_arg,
        snapshot_node,
        record_positions_node,
    ])
