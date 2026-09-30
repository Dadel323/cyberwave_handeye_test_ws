"""
intrinsics.launch.py

Webcam + intrinsics_node, for the one-off intrinsics calibration that has to
happen before any hand-eye run.

  ros2 launch handeye_calibration intrinsics.launch.py

Then, holding the ChArUco board in front of the camera, for each pose:
  ros2 service call /intrinsics_node/capture std_srvs/srv/Trigger
and finally:
  ros2 service call /intrinsics_node/calibrate std_srvs/srv/Trigger

The board parameters come from the same pipeline_params.yaml the hand-eye nodes
read, so both stages model the printed board identically.

Two parameter files are in play, and they are named apart on purpose:
  params_file           -- the camera's, owned by webcam.launch.py
                           (config/webcam_params.yaml)
  pipeline_params_file  -- intrinsics_node's board and output_path
                           (config/pipeline_params.yaml)
Sharing one name is what previously let this file's value reach usb_cam, which
silently reverted every camera parameter to its built-in default.

Only pipeline_params_file is declared below. Everything the camera takes --
video_device, params_file -- is declared by webcam.launch.py, and launch
configurations set on this command line reach an included file whether or not
the including file declares them. So `video_device:=/dev/video1` still works,
and there is one declaration of it rather than two that can disagree.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('handeye_calibration')

    args = [
        DeclareLaunchArgument(
            'pipeline_params_file',
            default_value=PathJoinSubstitution(
                [pkg_share, 'config', 'pipeline_params.yaml']),
            description="intrinsics_node's parameters: board geometry and the "
                        'camera_info path ~/calibrate writes.'),
    ]

    # No launch_arguments: webcam.launch.py declares video_device and
    # params_file itself, with its own defaults, and anything given on this
    # command line reaches it regardless. webcam.launch.py is the single place
    # the camera's configuration lives.
    webcam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, 'launch', 'webcam.launch.py'])),
    )

    intrinsics_node = Node(
        package='handeye_calibration',
        executable='intrinsics_node',
        name='intrinsics_node',
        output='screen',
        parameters=[LaunchConfiguration('pipeline_params_file')],
    )

    return LaunchDescription(args + [webcam, intrinsics_node])
