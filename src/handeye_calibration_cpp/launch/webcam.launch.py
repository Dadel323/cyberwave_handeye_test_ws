"""
webcam.launch.py

Brings up a plain UVC webcam as the calibration camera, publishing
/webcam/image_raw and /webcam/camera_info.

Autofocus MUST stay off: refocusing changes the focal length, so intrinsics
measured at one focus setting do not describe frames captured at another. That
would corrupt both the intrinsics solve and every board pose in the hand-eye
dataset.

usb_cam's own `autofocus: false` parameter cannot do this here -- it shells out
to `v4l2-ctl --set-ctrl=focus_auto=0`, and that control is named
`focus_automatic_continuous` on this kernel, so the call fails silently. The
apply_controls action below sets it under its current name and reads the values
back so a failure is visible.

Focus, exposure and their auto flags come from the `v4l2_controls` section of
the params file -- the only place they are set, so the intrinsics run and the
hand-eye run cannot drift apart.

To find the right values interactively, run camera_tuner.py at the workspace
root with nothing else holding the camera -- its "Print settings" button emits
a block ready to paste into webcam_params.yaml.
"""

import yaml

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            OpaqueFunction, TimerAction)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# A control gated by an "auto" flag is inactive, and writing it while the flag
# is on returns EACCES. These have to go first; anything else in the config
# follows in file order.
GATING_CONTROLS = ['focus_automatic_continuous', 'auto_exposure']


def apply_controls(context, *args, **kwargs):
    """Build one shell command that sets every configured V4L2 control.

    A single sequential command rather than one ExecuteProcess per control:
    launch starts sibling processes concurrently, which would race the gating
    flags against the controls they gate. Setting them all in one v4l2-ctl call
    does not work either -- that is a single VIDIOC_S_EXT_CTRLS, validated
    before the gating flag in the same batch has taken effect.
    """
    device = LaunchConfiguration('video_device').perform(context)
    params = yaml.safe_load(
        open(LaunchConfiguration('params_file').perform(context))) or {}
    controls = params.get('v4l2_controls', {}).get('ros__parameters', {})
    if not controls:
        return []

    names = ([c for c in GATING_CONTROLS if c in controls]
             + [c for c in controls if c not in GATING_CONTROLS])
    steps = [f'v4l2-ctl --device {device} --set-ctrl {n}={controls[n]}'
             f' || echo "  !! could not set {n}"' for n in names]
    steps.append(f'v4l2-ctl --device {device} --get-ctrl {",".join(names)}')
    return [ExecuteProcess(cmd=['bash', '-c', '; '.join(steps)], output='screen')]


def generate_launch_description():
    video_device = LaunchConfiguration('video_device')
    params_file = LaunchConfiguration('params_file')

    args = [
        DeclareLaunchArgument(
            'video_device', default_value='/dev/video0',
            description='Camera device node.'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('handeye_calibration_cpp'), 'config',
                 'webcam_params.yaml']),
            description="usb_cam's parameters, plus the v4l2_controls section "
                        'this file applies with v4l2-ctl.'),
    ]

    usb_cam = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        namespace='webcam',
        output='screen',
        parameters=[params_file, {'video_device': video_device}],
    )

    # Applied after the device is streaming -- UVC controls set on an open,
    # streaming device stick reliably, whereas some are reset at stream start.
    lock_controls = TimerAction(
        period=3.0, actions=[OpaqueFunction(function=apply_controls)])

    return LaunchDescription(args + [usb_cam, lock_controls])
