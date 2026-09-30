"""
viewer.launch.py

Live view of the calibration camera on top of webcam.launch.py -- the camera
with its V4L2 controls locked exactly as the intrinsics and hand-eye runs use
it, plus viewer_node showing the compressed stream.

Use it to check framing, focus and exposure against what the pipeline will
actually see. For a view alongside an already-running camera, run the viewer
alone instead:

    ros2 run handeye_calibration viewer_node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    video_device = LaunchConfiguration('video_device')
    image_topic = LaunchConfiguration('image_topic')

    args = [
        DeclareLaunchArgument(
            'video_device', default_value='/dev/video0',
            description='Camera device node, passed through to webcam.launch.py.'),
        DeclareLaunchArgument(
            'image_topic', default_value='/webcam/image_raw',
            description='Base image topic; the viewer subscribes to '
                        '<image_topic>/compressed.'),
    ]

    webcam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('handeye_calibration'), 'launch',
             'webcam.launch.py'])),
        launch_arguments={'video_device': video_device}.items(),
    )

    viewer = Node(
        package='handeye_calibration',
        executable='viewer_node',
        name='viewer_node',
        output='screen',
        parameters=[{'image_topic': image_topic}],
    )

    return LaunchDescription(args + [webcam, viewer])
