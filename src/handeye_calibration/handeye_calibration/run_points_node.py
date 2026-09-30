#!/usr/bin/env python3
"""
run_points_node.py

Exposes ~/move_to_next_point (handeye_calibration_interfaces/srv/MoveToNextPoint).

Each call advances to the NEXT joint-space pose loaded from manifest.yaml
and moves the robot there (one pose per call).

Bypasses MoveIt/IK entirely -- publishes straight to
/so101_follower/joint_commands, which motor_bridge already subscribes
to and drives via its 50 Hz loop. Requires motor_bridge to already be
running with torque enabled (its normal resting state whenever a
move_to_next_point call isn't active).

Usage:
    ros2 run handeye_calibration run_points_node --ros-args \
        -p recorded_points_path:=/home/cyb/projects/handeye_ws/src/record_positions/data/manifest.yaml

Then, from another terminal:

    ros2 service call /run_points_node/move_to_next_point \
        handeye_calibration_interfaces/srv/MoveToNextPoint "{}"
"""

import time
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from handeye_calibration_interfaces.srv import MoveToNextPoint



class RunPointsNode(Node):
    def __init__(self):
        super().__init__('run_points_node')

        self.declare_parameter('recorded_points_path', '/home/cyb/projects/ros_ws/src/record_positions/data/manifest.yaml')
        self.declare_parameter('commands_topic', '/so101_follower/joint_commands')
        self.declare_parameter('joint_state_topic', '/so101_follower/joint_states')
        
        self.declare_parameter('use_joint_command', False)
        self.declare_parameter('trajectory_duration_sec', 1.5)

        joint_state_topic = self.get_parameter('joint_state_topic').value
        recorded_points_path = self.get_parameter('recorded_points_path').value
        topic = self.get_parameter('commands_topic').value
        self.use_joint_command = self.get_parameter('use_joint_command').value
        self.joint_states_dict = None
        self.trajectory_duration_sec = self.get_parameter('trajectory_duration_sec').value
        self.poses = []
        self._next_index = 0
        if recorded_points_path:
            self.poses = self._load_joint_poses(recorded_points_path)
            self.get_logger().info(f'Loaded {len(self.poses)} joint-space poses from {recorded_points_path}')
        else:
            self.get_logger().warn(
                'No recorded_points_path given -- move_to_next_point calls '
                'will have nothing to advance through.'
            )

        if not self.use_joint_command:
            self._traj_pub = self.create_publisher(JointTrajectory, topic, 10)
        else:
            self._pub = self.create_publisher(JointState, topic, 10)
        
        self.create_service(MoveToNextPoint, '~/move_to_next_point', self._srv_move_to_next_point)

        self.create_subscription(JointState, joint_state_topic, self._joint_state_cb, 10)

        self.get_logger().info('run_points_node ready. Service: ~/move_to_next_point')
        
    
    def _joint_state_cb(self, msg: JointState):
        # This callback can be used to monitor the current joint states if needed.
        # For now, we just log the received joint states for debugging purposes.
        self.get_logger().debug(f'Received joint states: {msg.name} -> {msg.position}')
        self.joint_states_dict = dict(zip(msg.name, msg.position))

    def _load_joint_poses(self, recorded_points_path):
        with open(recorded_points_path, 'r', encoding='utf-8') as fp:
            manifest = yaml.safe_load(fp)

        poses = []
        skipped = 0
        for entry in manifest.get('entries', []):
            joint_positions = entry.get('joint_positions')
            if not joint_positions:
                skipped += 1
                continue
            poses.append(joint_positions)

        if skipped:
            self.get_logger().warn(
                f'{skipped} entries had no joint_positions recorded, skipped.'
            )
        return poses
    
    def _publish_joint_positions(self, joint_positions: dict):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_positions.keys())
        msg.position = list(joint_positions.values())
        self._pub.publish(msg)
        # Block until the arm has arrived, exactly as the trajectory path below
        # does. Without this the service returns while the joints are still
        # travelling, and capture_node -- which is called the moment it returns
        # -- pairs a motion-blurred frame with a TF pose from the wrong instant.
        # Nothing errors; the samples are simply wrong.
        #
        # This sleep is a BLOCKING call in a service callback on a
        # single-threaded executor, so it stalls every other callback in this
        # node for its whole duration. The two logs below bracket that stall so
        # it is visible in the timeline.
        nap = self.trajectory_duration_sec + 1.0
        self.get_logger().info(
            f'    [joint_command] published; SLEEP {nap:.2f}s START '
            '(executor blocked -- no callbacks run in this node)')
        t0 = time.monotonic()
        time.sleep(nap)
        self.get_logger().info(
            f'    [joint_command] SLEEP END after {time.monotonic() - t0:.3f}s '
            '(executor released)')
        
    def _publish_joint_trajectory(self, joint_positions: dict) -> bool:
        if self._traj_pub is None:
            self.get_logger().error('control_mode is joint_trajectory but publisher was not created.')
            return False
 
        duration_sec = self.trajectory_duration_sec
 
        msg = JointTrajectory()
        msg.joint_names = list(joint_positions.keys())
 
        point = JointTrajectoryPoint()
        point.positions = list(joint_positions.values())
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points = [point]
 
        self._traj_pub.publish(msg)
        # Blocking, same executor-stalling caveat as _publish_joint_positions.
        nap = duration_sec + 1.0
        self.get_logger().info(
            f'    [joint_trajectory] published; SLEEP {nap:.2f}s START '
            '(executor blocked -- no callbacks run in this node)')
        t0 = time.monotonic()
        time.sleep(nap)  # Wait for the trajectory to complete
        self.get_logger().info(
            f'    [joint_trajectory] SLEEP END after {time.monotonic() - t0:.3f}s '
            '(executor released)')

    # ------------------------------------------------------------
    def _srv_move_to_next_point(self, request, response):
        srv_t0 = time.monotonic()
        self.get_logger().info('>>> move_to_next_point ENTER')

        if self._next_index >= len(self.poses):
            response.success = False
            response.message = (
                f'No more recorded poses ({len(self.poses)} total, all used).'
            )
            response.joint_names = []
            response.joint_positions = []
            self.get_logger().warn(response.message)
            self.get_logger().info(
                f'<<< move_to_next_point EXIT (took {time.monotonic() - srv_t0:.3f}s) '
                '-- no move performed')
            return response

        joint_positions = self.poses[self._next_index]
        self._next_index += 1
        self.get_logger().info(
            f'[recorded {self._next_index}/{len(self.poses)}] Moving to: {joint_positions}'
        )

        
        #self._publish_smooth_motion(joint_positions, duration_sec=1.0)
        if not self.use_joint_command:
            self._publish_joint_trajectory(joint_positions)
        else:
            self._publish_joint_positions(joint_positions)

        response.success = True
        response.message = f'Moved to recorded pose'
        response.joint_names = list(joint_positions.keys())
        response.joint_positions = list(joint_positions.values())
        self.get_logger().info(
            f'<<< move_to_next_point EXIT (took {time.monotonic() - srv_t0:.3f}s) '
            '-- arm settled; sample_position is called next')
        return response

def main():
    rclpy.init()
    node = RunPointsNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()