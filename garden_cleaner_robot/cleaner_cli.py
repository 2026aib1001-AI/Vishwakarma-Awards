#!/usr/bin/env python3
"""
Interactive Terminal Mission Control CLI for Garden Cleaner Robot.

Allows one-keystroke command dispatch:
  [c] Clean Path C to A
  [a] Clean Path A to B
  [b] Clean Path B to C
  [m] Clean All (Complete Triangle Sweep)
  [1] Go to Zone A (South Plaza)
  [2] Go to Zone B (East Pavilion)
  [3] Go to Zone C (West Fountain)
  [p] Resume Loop Patrol
  [s] Stop / Pause Robot
  [q] Quit CLI
"""

import sys
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry


class CleanerCLI(Node):
    def __init__(self):
        super().__init__('cleaner_cli')

        self.score = 0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.nav_status = "Standby"
        self.gps_info = "Acquiring Fix..."

        self.goto_pub = self.create_publisher(String, '/garden/go_to', 10)

        self.create_subscription(Int32, '/garden/score', self.score_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(String, '/garden/nav_status', self.nav_cb, 10)
        self.create_subscription(String, '/garden/gps_status', self.gps_cb, 10)

        self.print_menu()

    def score_cb(self, msg: Int32):
        self.score = msg.data

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def nav_cb(self, msg: String):
        self.nav_status = msg.data

    def gps_cb(self, msg: String):
        self.gps_info = msg.data

    def send_cmd(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.goto_pub.publish(msg)
        print(f"\n>> Sent Command: '{cmd}'")

    def print_menu(self):
        menu = f"""
╔══════════════════════════════════════════════════════════════╗
║        🌿 GARDEN CLEANER ROBOT - MISSION COMMAND CLI 🌿      ║
╠══════════════════════════════════════════════════════════════╣
║  [e] Autonomous Frontier Explore   [m] Clean All (Triangle)  ║
║  [c] Clean Path C to A             [p] Resume Loop Patrol    ║
║  [a] Clean Path A to B             [s] Stop / Pause Robot    ║
║  [b] Clean Path B to C             [f] Go to Water Fountain  ║
║  [1] Go to Zone A (South Plaza)    [t] Go to Park Bench      ║
║  [2] Go to Zone B (East Pavilion)  [q] Quit CLI              ║
║  [3] Go to Zone C (West Fountain)                            ║
╠══════════════════════════════════════════════════════════════╣
║  Live Score: {self.score:2d}/60 PTS   | Pos: ({self.robot_x:+.2f}, {self.robot_y:+.2f})             ║
║  GPS: {self.gps_info[:52]:<52}   ║
║  Status: {self.nav_status[:48]:<48}    ║
╚══════════════════════════════════════════════════════════════╝
Command (press key): """
        print(menu, end='', flush=True)


def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = CleanerCLI()

    key_map = {
        'e': 'explore',
        'c': 'clean c to a',
        'a': 'clean a to b',
        'b': 'clean b to c',
        'm': 'clean all',
        '1': 'go to a',
        '2': 'go to b',
        '3': 'go to c',
        'p': 'patrol',
        's': 'stop',
        'f': 'fountain',
        't': 'bench',
    }

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            key = getKey(settings)
            if key == 'q' or key == '\x03':
                break
            elif key in key_map:
                node.send_cmd(key_map[key])
                node.print_menu()
    except Exception as e:
        print(e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
