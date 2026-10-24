#!/usr/bin/env python3
"""
Dynamic Pedestrian Simulator for Garden Park.

Drives the walking human model in Gazebo Harmonic via /pedestrian/cmd_vel:
- Paces along the North Path (B->C) between X = -3.0 and X = +3.0 at Y = 6.0.
- Pauses at the pathway center to simulate natural park visitor movement.
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped


class PedestrianSimulator(Node):
    def __init__(self):
        super().__init__('pedestrian_simulator')

        self.declare_parameter('walking_speed', 0.45)
        self.speed = self.get_parameter('walking_speed').get_parameter_value().double_value

        self.cmd_pub = self.create_publisher(Twist, '/pedestrian/cmd_vel', 10)
        self.human_pub = self.create_publisher(PoseStamped, '/detected_human', 10)
        self.gt_pose_pub = self.create_publisher(PoseStamped, '/pedestrian/pose', 10)

        # States: 'WALKING_WEST', 'PAUSE_WEST', 'WALKING_EAST', 'PAUSE_EAST'
        self.state = 'WALKING_WEST'
        self.pos_x = 0.0
        self.pause_time = 0.0
        self.last_time = time.time()

        self.timer = self.create_timer(0.05, self.update_loop)
        self.get_logger().info('🚶 Garden Pedestrian Simulator pacing on Path B->C.')

    def update_loop(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        twist = Twist()

        if self.state == 'WALKING_WEST':
            twist.linear.x = -self.speed
            self.pos_x -= self.speed * dt
            if self.pos_x <= -2.8:
                self.pos_x = -2.8
                self.state = 'PAUSE_WEST'
                self.pause_time = now
                self.get_logger().info('🚶 Pedestrian arrived near West Fountain, pausing...')

        elif self.state == 'PAUSE_WEST':
            twist.linear.x = 0.0
            if now - self.pause_time > 3.0:
                self.state = 'WALKING_EAST'
                self.get_logger().info('🚶 Pedestrian turning east toward Pavilion...')

        elif self.state == 'WALKING_EAST':
            twist.linear.x = self.speed
            self.pos_x += self.speed * dt
            if self.pos_x >= 2.8:
                self.pos_x = 2.8
                self.state = 'PAUSE_EAST'
                self.pause_time = now
                self.get_logger().info('🚶 Pedestrian arrived near East Pavilion, pausing...')

        elif self.state == 'PAUSE_EAST':
            twist.linear.x = 0.0
            if now - self.pause_time > 3.0:
                self.state = 'WALKING_WEST'
                self.get_logger().info('🚶 Pedestrian strolling west through the park...')

        self.cmd_pub.publish(twist)

        # Publish dynamic human pose for sensory avoidance & visualizers
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = self.pos_x
        pose_msg.pose.position.y = 6.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        self.human_pub.publish(pose_msg)
        self.gt_pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
