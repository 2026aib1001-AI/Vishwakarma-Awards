#!/usr/bin/env python3
"""
SLAM Toolbox Lifecycle Activator for Garden Cleaner Robot.

Ensures async_slam_toolbox_node transitions to 'active' [3] reliably.
Periodically checks /slam_toolbox/get_state, and if unconfigured or inactive,
calls /slam_toolbox/change_state to configure and activate it.
Exits once the node reaches 'active'.
"""

import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition, State


class SlamLifecycleActivator(Node):
    def __init__(self):
        super().__init__('slam_lifecycle_activator')
        self.change_client = self.create_client(ChangeState, '/slam_toolbox/change_state')
        self.get_client = self.create_client(GetState, '/slam_toolbox/get_state')
        self.timer = self.create_timer(1.0, self.check_and_transition)
        self.attempts = 0
        self.get_logger().info('🧭 SLAM Lifecycle Activator: Monitoring /slam_toolbox state...')

    def check_and_transition(self):
        self.attempts += 1
        if self.attempts > 40:
            self.get_logger().warning('Timed out waiting for SLAM Toolbox lifecycle.')
            rclpy.shutdown()
            return

        if not self.get_client.service_is_ready():
            return

        req = GetState.Request()
        future = self.get_client.call_async(req)
        future.add_done_callback(self.state_callback)

    def state_callback(self, future):
        try:
            res = future.result()
            state_id = res.current_state.id
            state_label = res.current_state.label
            self.get_logger().info(f'SLAM Toolbox state: {state_label} [{state_id}]')

            if state_id == State.PRIMARY_STATE_ACTIVE:
                self.get_logger().info('✅ SLAM Toolbox is ACTIVE! /map is now available.')
                self.timer.cancel()
                rclpy.shutdown()
                return

            if state_id == State.PRIMARY_STATE_UNCONFIGURED:
                self.get_logger().info('Configuring SLAM Toolbox (transition 1)...')
                change_req = ChangeState.Request()
                change_req.transition.id = Transition.TRANSITION_CONFIGURE
                self.change_client.call_async(change_req)

            elif state_id == State.PRIMARY_STATE_INACTIVE:
                self.get_logger().info('Activating SLAM Toolbox (transition 3)...')
                change_req = ChangeState.Request()
                change_req.transition.id = Transition.TRANSITION_ACTIVATE
                self.change_client.call_async(change_req)

        except Exception as e:
            self.get_logger().error(f'Failed to check/change SLAM state: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SlamLifecycleActivator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
