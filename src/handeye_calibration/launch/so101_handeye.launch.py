"""
so101_handeye.launch.py

The unattended half of the hand-eye rig for the SO-101: robot_state_publisher
for TF, the camera (webcam.launch.py), and the two handeye nodes that only ever
answer service calls (capture_node, calibration_node). All read
config/so101_params.yaml.

Two processes are deliberately NOT started here, because both need a terminal
on stdin and a process started by ros2 launch has none -- the prompt would
never return and the run would hang with no error. Each needs its own
terminal:

  # 1. this file
  ros2 launch handeye_calibration so101_handeye.launch.py

  # 2. the arm. Owns the Feetech bus, and prompts three times per sample in
  #    ~/move_to_next_point: relax, lock, return.
  ros2 run so101_hw_interface so101_motor_bridge --ros-args \
      --params-file <pkg>/config/so101_params.yaml

  # 3. the orchestrator
  ros2 run handeye_calibration auto_calibration_node --ros-args \
      --params-file <pkg>/config/so101_params.yaml

Per sample the orchestrator calls the bridge's ~/move_to_next_point (via its
point_sampler_node parameter, in place of run_points_node), then capture_node's
~/sample_position for the frame and the TF lookup. After the last sample it
calls calibrate_dataset and writes handeye_result.yaml.

robot_state_publisher lives here rather than with the bridge, but it is driven
by the bridge's joint_states -- so TF only becomes live once terminal 2 runs.

Intrinsics must already exist -- run intrinsics.launch.py once first.
Until config/webcam_intrinsics.yaml is written, usb_cam publishes an all-zero
CameraInfo and capture_node rejects every sample.

Two parameter files are in play, and they are named apart on purpose:
  params_file           -- the camera's, owned by webcam.launch.py
                           (config/webcam_params.yaml)
  pipeline_params_file  -- the arm's, the board's and the frames'
                           (config/so101_params.yaml)
Sharing one name does not merely shadow a default: a launch configuration set
here is visible inside an included file, and the child's own
DeclareLaunchArgument does not override it. Calling this one `params_file` sent
so101_params.yaml to usb_cam, which matched none of its keys and so silently
reverted every camera parameter to its built-in default -- 640x480 yuyv rather
than 1280x720 mjpeg2rgb, no camera_info_url, and no v4l2_controls section for
webcam.launch.py to apply.

video_device and params_file are declared by webcam.launch.py, not here, for
the same reason turned around: configurations set on this command line reach an
included file regardless, so `video_device:=/dev/video1` and
`params_file:=<other webcam yaml>` still work, and each is declared once rather
than twice where the two could disagree.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('handeye_calibration')
    default_params = os.path.join(pkg_share, 'config', 'so101_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'pipeline_params_file', default_value=default_params,
        description='Arm, board, frames and output_dir for every node here. '
                    "Named apart from the camera's params_file -- see above.")
    params_file = LaunchConfiguration('pipeline_params_file')

    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'), ' ',
            PathJoinSubstitution([
                FindPackageShare('so101_follower_description'),
                'urdf', 'so101_follower.urdf.xacro']),
        ]),
        value_type=str)

    # TF for capture_node's base_link -> gripper_link lookup. Driven by the
    # joint_states motor_bridge publishes, hence the remap -- the bridge
    # namespaces them under the arm rather than using the global topic.
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
        remappings=[('/joint_states', '/so101_follower/joint_states')],
    )

    # No launch_arguments: webcam.launch.py declares video_device and
    # params_file itself, with its own defaults, and anything given on this
    # command line reaches it regardless. webcam.launch.py is the single place
    # the camera's configuration lives.
    webcam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('handeye_calibration'),
                'launch', 'webcam.launch.py'])),
    )

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
    
    orchistrator_node = Node(
        package='handeye_calibration',
        executable='auto_calibration_node',
        name='auto_calibration_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        params_file_arg,
        robot_state_publisher,
        webcam,
        capture_node,
        calibration_node,
        orchistrator_node
    ])
