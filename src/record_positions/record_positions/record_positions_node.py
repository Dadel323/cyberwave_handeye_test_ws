import sys

import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from handeye_calibration_interfaces.srv import MoveToNextPoint


class RecordPositionsNode(Node):
    """Drives the teaching pipeline for a preset number of poses: for each
    pose, either calls a service to move to the next point (motor_bridge_ns)
    or prompts the user to reposition the arm by hand, then calls a service
    to record the current joint-space position into the manifest."""

    def __init__(self):
        super().__init__('record_positions_node')

        self.declare_parameter('motor_bridge_ns', None)
        self.declare_parameter('snapshot_node', '/snapshot_node')
        self.declare_parameter('num_samples', 30)

        self.num_samples = self.get_parameter('num_samples').value

        self.ns = self.get_parameter('motor_bridge_ns').value
        if self.ns is not None:
            self._move_client = self.create_client(MoveToNextPoint, f'{self.ns}/move_to_next_point')

        snapshot_ns = self.get_parameter('snapshot_node').value
        self._record_client = self.create_client(Trigger, f'{snapshot_ns}/record_position')

    # ------------------------------------------------------------
    def record_one(self, sample_num: int, total: int) -> bool:
        if self.ns is None:
            print("Reposition by hand, then press <Enter> to continue...")
            input()
            self.get_logger().info('Arm locked. Recording position...')
        else:
            self.get_logger().info(f'[{sample_num}/{total}] Calling move_to_next_point...')
            future = self._move_client.call_async(MoveToNextPoint.Request())
            rclpy.spin_until_future_complete(self, future)  # top-level spin, not nested -- safe

            response = future.result()
            if response is None:
                self.get_logger().error(f'[{sample_num}/{total}] move_to_next_point call failed (no response).')
                return False
            if not response.success:
                self.get_logger().error(f'[{sample_num}/{total}] move_to_next_point reported failure: {response.message}')
                return False

        future = self._record_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None:
            self.get_logger().error(f'[{sample_num}/{total}] record_position call failed (no response).')
            return False
        if not response.success:
            self.get_logger().error(f'[{sample_num}/{total}] record_position reported failure: {response.message}')
            return False

        self.get_logger().info(f'[{sample_num}/{total}] {response.message}')
        return True

    def record_many(self, num_samples: int) -> int:
        """Calls record_one() num_samples times in a row, stopping
        early (without erroring the whole run) if a call fails --
        e.g. motor_bridge going away mid-run. Returns the number of
        samples successfully recorded."""
        if self.ns is not None:
            self.get_logger().info('Waiting for motor_bridge move_to_next_point service...')
            if not self._move_client.wait_for_service(timeout_sec=10.0):
                self.get_logger().error('motor_bridge move_to_next_point service not available -- is motor_bridge running?')
                return 0

        self.get_logger().info('Waiting for record_position service...')
        if not self._record_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('record_position service not available -- is snapshot_node running?')
            return 0

        recorded = 0
        for i in range(1, num_samples + 1):
            if not self.record_one(i, num_samples):
                self.get_logger().error(
                    f'Stopping after {recorded}/{num_samples} samples due to failure above.'
                )
                break
            recorded += 1

        return recorded


def main():
    rclpy.init()
    node = RecordPositionsNode()
    try:
        recorded = node.record_many(node.num_samples)
        node.get_logger().info(f'Done: {recorded}/{node.num_samples} pose(s) recorded.')
        success = recorded == node.num_samples
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
