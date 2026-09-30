"""
publish_handeye_tf.launch.py

Publishes the solved hand-eye result as a static TF, gripper -> camera.

The transform is read out of handeye_result.yaml rather than written into this
file, so re-running the calibration is enough to change what gets published --
there is no second copy of the numbers to forget to update. The frame names
come from the same file, which is why they are not arguments here:
calibration_node wrote them from the parent_frame/child_frame it actually
solved for, and overriding them here could only ever disagree with the data.

calibrateHandEye returns R_cam2gripper/t_cam2gripper, and calibration_node
stores that as parent_frame=gripper_link, child_frame=webcam_optical_frame --
already the parent->child direction TF wants, so nothing is inverted below.

  ros2 launch handeye_calibration publish_handeye_tf.launch.py
  ros2 launch handeye_calibration publish_handeye_tf.launch.py \
      result_file:=/path/to/another/handeye_result.yaml

Verify once it is up:
  ros2 run tf2_ros tf2_echo gripper_link webcam_optical_frame
"""

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    result_file = LaunchConfiguration('result_file').perform(context)
    with open(result_file) as f:
        r = yaml.safe_load(f)

    t, q = r['translation'], r['rotation_xyzw']

    # static_transform_publisher takes these as string arguments, not
    # parameters. repr() rather than str() so the full double precision
    # survives -- the translation is in metres, so a truncated decimal is a
    # real millimetre-scale error in the camera position.
    return [Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='handeye_static_tf',
        output='screen',
        arguments=[
            '--x', repr(t['x']), '--y', repr(t['y']), '--z', repr(t['z']),
            '--qx', repr(q['x']), '--qy', repr(q['y']),
            '--qz', repr(q['z']), '--qw', repr(q['w']),
            '--frame-id', r['parent_frame'],
            '--child-frame-id', r['child_frame'],
        ],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'result_file',
            default_value='/home/cyb/projects/handeye_ws/src/handeye_calibration/'
                          'data_so101/handeye_result.yaml',
            description='handeye_result.yaml to publish. Supplies the '
                        'transform and both frame names.'),
        OpaqueFunction(function=launch_setup),
    ])
