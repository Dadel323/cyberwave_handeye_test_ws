#!/usr/bin/env python3
"""
snapshot_node.py

Exposes ~/record_position (std_srvs/srv/Trigger).

Each call takes whatever /joint_states last delivered, filters it down to
joints_to_keep, appends it as a new entry to manifest.yaml, and writes the
file. Single check, no waiting for a new message -- the orchestrator
(record_positions_node) is expected to have already moved/repositioned the
arm and given it a moment to settle before calling this.
"""

import pathlib

import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


class SnapshotNode(Node):
    def __init__(self):
        super().__init__('snapshot_node')

        self.declare_parameter(
            'manifest_path',
            '/home/cyb/projects/cyberwave_handeye_test_ws/src/record_positions/data/manifest.yaml',
        )
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('overwrite', True)
        self.declare_parameter('joints_to_keep', [
            'openarm_right_joint1', 'openarm_right_joint2', 'openarm_right_joint3',
            'openarm_right_joint4', 'openarm_right_joint5', 'openarm_right_joint6',
            'openarm_right_joint7',
        ])

        self.manifest_path = pathlib.Path(self.get_parameter('manifest_path').value).expanduser()
        self.joints_to_keep = list(self.get_parameter('joints_to_keep').value)

        self._latest_joint_state = None  # dict: name -> position
        self.create_subscription(
            JointState, self.get_parameter('joint_states_topic').value,
            self._joint_state_cb, 10)

        self.manifest = self._load_or_init_manifest(self.get_parameter('overwrite').value)

        self.create_service(Trigger, '~/record_position', self._srv_record_position)

        self.get_logger().info('snapshot_node ready. Service: ~/record_position')

    # ------------------------------------------------------------
    # Manifest handling
    # ------------------------------------------------------------
    def _load_or_init_manifest(self, overwrite: bool) -> dict:
        if self.manifest_path.exists() and not overwrite:
            with open(self.manifest_path, 'r', encoding='utf-8') as fp:
                loaded = yaml.safe_load(fp) or {}
            loaded.setdefault('entries', [])
            self.get_logger().info(
                f'Appending to existing manifest ({len(loaded["entries"])} entries found).'
            )
            return loaded

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists() and overwrite:
            self.get_logger().warn(f'overwrite:=true -- starting a fresh manifest at {self.manifest_path}')
        return {'entries': []}

    def _write_manifest(self):
        with open(self.manifest_path, 'w', encoding='utf-8') as fp:
            yaml.safe_dump(self.manifest, fp, sort_keys=False)

    # ------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState):
        self._latest_joint_state = dict(zip(msg.name, msg.position))

    def _filter_joints(self, joint_positions: dict) -> dict:
        """Applies joints_to_keep, if set. Warns (but doesn't fail) if
        a requested joint name isn't present in the snapshot -- e.g. a
        typo, or the wrong arm's joint names -- since silently dropping
        an entire sample over a name mismatch would be worse than
        surfacing it loudly here."""
        if not self.joints_to_keep:
            return joint_positions

        filtered = {}
        missing = []
        for name in self.joints_to_keep:
            if name in joint_positions:
                filtered[name] = joint_positions[name]
            else:
                missing.append(name)

        if missing:
            self.get_logger().warn(
                f'joints_to_keep requested {missing}, but these were not present '
                f'in the /joint_states snapshot (available: {list(joint_positions.keys())}). '
                f'Check for a typo or a mismatched joint_states_topic.'
            )
        return filtered

    # ------------------------------------------------------------
    def _srv_record_position(self, request, response):
        if self._latest_joint_state is None:
            response.success = False
            response.message = 'No /joint_states message received yet.'
            self.get_logger().error(response.message)
            return response

        joint_positions = self._filter_joints(dict(self._latest_joint_state))
        if not joint_positions:
            response.success = False
            response.message = 'joints_to_keep filtering left an empty dict -- not saving this sample.'
            self.get_logger().error(response.message)
            return response

        self.manifest['entries'].append({'joint_positions': joint_positions})
        self._write_manifest()

        idx = len(self.manifest['entries'])
        response.success = True
        response.message = f'Saved pose {idx} -> {self.manifest_path}'
        self.get_logger().info(f'{response.message}: {joint_positions}')
        return response


def main():
    rclpy.init()
    node = SnapshotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
