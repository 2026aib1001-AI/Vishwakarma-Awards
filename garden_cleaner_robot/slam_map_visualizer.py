#!/usr/bin/env python3
"""
Real-Time SLAM Map Pop-Up Visualizer with AI Vision Litter Pins & Mission HUD.
Zero Pre-Determined Locations: Map and HUD dynamically populate litter items
strictly when discovered by the onboard AI computer vision detector.

Features:
1. Dual-Panel Desktop Window:
   - Left Panel (720x720): Crisp 2D Occupancy Grid SLAM map with trajectory trail, planned path,
     robot heading [R], and dynamically discovered litter pins.
   - Right Panel (300x720): Dedicated Telemetry & Mission HUD Sidebar with Score, GPS, Speed,
     Hopper Capacity, and AI Vision Discovery Feed.
2. Dynamic Discovery Pins:
   - Red Cans rendered as crimson pins.
   - Blue Cans rendered as azure pins.
   - Yellow Boxes rendered as golden pins.
   - Collected items fade to clean green checkmark pins [v].
   - Robot rendered as cyan badge [R] with directional heading arrow.
3. 100% Native Desktop OpenCV GUI Window at 12 Hz.
"""

import os
import math
import time
import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Int32, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener


class SlamMapVisualizer(Node):
    def __init__(self):
        super().__init__('slam_map_visualizer')

        self.declare_parameter('window_size', 720)
        self.declare_parameter('show_window', True)
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.show_window = self.get_parameter('show_window').get_parameter_value().bool_value

        self.map_w = self.window_size
        self.map_h = self.window_size
        self.sidebar_w = 300
        self.total_w = self.map_w + self.sidebar_w
        self.total_h = self.map_h

        self.gui_available = bool(os.environ.get('DISPLAY')) and self.show_window
        if self.gui_available:
            try:
                cv2.namedWindow('Garden Cleaner - Live SLAM Map & Mission HUD', cv2.WINDOW_AUTOSIZE)
                self.get_logger().info('Native SLAM Map Pop-Up Window initialized with HUD Sidebar.')
            except Exception as e:
                self.get_logger().warning(f'Could not initialize OpenCV map window: {e}')
                self.gui_available = False

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State Cache
        self.latest_map = None
        self.latest_path = []
        self.latest_trajectory = []
        self.status_text = "INITIALIZING SLAM"
        self.gps_text = "GPS: Waiting for fix..."
        self.score = 0
        self.bin_capacity_pct = 0.0
        self.robot_pose = None  # (x, y, yaw)
        self.current_cmd_v = 0.0
        self.current_cmd_w = 0.0

        # ZERO Prior Knowledge: Discovered dynamically via AI Vision
        self.discovered_trash = {}  # key -> dict(id, name, x, y, label, color, collected, time)

        # Simulation world ground-truth targets (maintained for compatibility/referee testing)
        self.trash_targets = {
            'trash_a1': {'name': 'Red Can A1', 'path': 'A->B', 'x': 2.5, 'y': -3.2, 'color': (30, 30, 230), 'collected': False},
            'trash_a2': {'name': 'Red Can A2', 'path': 'A->B', 'x': 7.2, 'y': 2.8, 'color': (30, 30, 230), 'collected': False},
            'trash_b1': {'name': 'Blue Can B1', 'path': 'B->C', 'x': 4.0, 'y': 6.0, 'color': (230, 140, 20), 'collected': False},
            'trash_b2': {'name': 'Blue Can B2', 'path': 'B->C', 'x': -4.0, 'y': 6.0, 'color': (230, 140, 20), 'collected': False},
            'trash_c1': {'name': 'Yellow Box C1', 'path': 'C->A', 'x': -7.2, 'y': 2.8, 'color': (20, 215, 240), 'collected': False},
            'trash_c2': {'name': 'Yellow Box C2', 'path': 'C->A', 'x': -2.5, 'y': -3.2, 'color': (20, 215, 240), 'collected': False},
        }

        # QoS for latched topics
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Subscriptions
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, latched_qos)
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_callback, 10)
        self.traj_sub = self.create_subscription(
            Path, '/robot_trajectory', self.trajectory_callback, 10)
        self.status_sub = self.create_subscription(
            String, '/garden/nav_status', self.status_callback, 10)
        self.score_sub = self.create_subscription(
            Int32, '/garden/score', self.score_callback, 10)
        self.bin_cap_sub = self.create_subscription(
            Float32, '/garden/bin_capacity_pct', self.bin_cap_callback, 10)
        self.gps_sub = self.create_subscription(
            String, '/garden/gps_status', self.gps_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)

        # Dynamic Vision & Collection Subscriptions
        self.detected_trash_sub = self.create_subscription(
            PoseStamped, '/garden/detected_trash', self.detected_trash_callback, 10)
        self.vision_json_sub = self.create_subscription(
            String, '/garden/vision_detections', self.vision_json_callback, 10)
        self.trash_collected_sub = self.create_subscription(
            String, '/garden/trash_collected', self.trash_collected_callback, 10)

        # Publisher for image stream
        self.map_image_pub = self.create_publisher(Image, '/garden/map_image', 5)

        # Render loop at 12 Hz
        self.render_timer = self.create_timer(0.08, self.render_loop)
        self.get_logger().info('=== Garden Cleaner SLAM Map Visualizer Active (Zero Prior Knowledge) ===')

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def path_callback(self, msg: Path):
        self.latest_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def trajectory_callback(self, msg: Path):
        self.latest_trajectory = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def status_callback(self, msg: String):
        self.status_text = msg.data

    def score_callback(self, msg: Int32):
        self.score = msg.data

    def bin_cap_callback(self, msg: Float32):
        self.bin_capacity_pct = msg.data

    def gps_callback(self, msg: String):
        self.gps_text = msg.data

    def cmd_callback(self, msg: Twist):
        self.current_cmd_v = msg.linear.x
        self.current_cmd_w = msg.angular.z

    def detected_trash_callback(self, msg: PoseStamped):
        """Dynamically registers litter when spotted by the AI computer vision detector."""
        tx = msg.pose.position.x
        ty = msg.pose.position.y

        # Check if already registered within 0.7m
        for item in self.discovered_trash.values():
            if math.hypot(tx - item['x'], ty - item['y']) < 0.70:
                return

        # Determine class and visual color
        class_str = "Litter Item"
        if '|' in msg.header.frame_id:
            class_str = msg.header.frame_id.split('|')[1].strip()

        color = (0, 200, 255)
        if 'RED' in class_str.upper():
            color = (30, 30, 235)  # Crimson
            class_str = "Red Soda Can"
        elif 'BLUE' in class_str.upper():
            color = (235, 130, 20)  # Azure
            class_str = "Blue Drink Can"
        elif 'YELLOW' in class_str.upper() or 'BOX' in class_str.upper():
            color = (20, 215, 240)  # Amber
            class_str = "Yellow Snack Box"

        item_idx = len(self.discovered_trash) + 1
        item_key = f"litter_{item_idx}"
        self.discovered_trash[item_key] = {
            'id': item_key,
            'name': f"Item #{item_idx} ({class_str})",
            'label': class_str,
            'x': tx,
            'y': ty,
            'color': color,
            'collected': False,
            'time': time.time()
        }
        self.get_logger().info(f"✨ [NEW DISCOVERY] {class_str} dynamically registered at ({tx:.1f}, {ty:.1f})!")

    def vision_json_callback(self, msg: String):
        """Parses multi-object detections from vision detector."""
        try:
            dets = json.loads(msg.data)
            for d in dets:
                tx = d.get('map_x', 0.0)
                ty = d.get('map_y', 0.0)
                cls_name = d.get('class', 'Litter Item')

                matched = False
                for item in self.discovered_trash.values():
                    if math.hypot(tx - item['x'], ty - item['y']) < 0.70:
                        matched = True
                        break

                if not matched:
                    color = (0, 200, 255)
                    label = cls_name
                    if 'RED' in cls_name.upper():
                        color = (30, 30, 235)
                        label = "Red Soda Can"
                    elif 'BLUE' in cls_name.upper():
                        color = (235, 130, 20)
                        label = "Blue Drink Can"
                    elif 'YELLOW' in cls_name.upper() or 'BOX' in cls_name.upper():
                        color = (20, 215, 240)
                        label = "Yellow Snack Box"

                    item_idx = len(self.discovered_trash) + 1
                    item_key = f"litter_{item_idx}"
                    self.discovered_trash[item_key] = {
                        'id': item_key,
                        'name': f"Item #{item_idx} ({label})",
                        'label': label,
                        'x': tx,
                        'y': ty,
                        'color': color,
                        'collected': False,
                        'time': time.time()
                    }
        except Exception:
            pass

    def trash_collected_callback(self, msg: String):
        """Called when vacuum collects a piece of litter."""
        try:
            parts = msg.data.split(':')
            t_name = parts[0]
            if t_name in self.trash_targets:
                self.trash_targets[t_name]['collected'] = True
            if len(parts) >= 3:
                cx = float(parts[1])
                cy = float(parts[2])
                for item in self.discovered_trash.values():
                    if math.hypot(cx - item['x'], cy - item['y']) < 0.80:
                        item['collected'] = True
                        break
        except Exception:
            pass

    def update_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            rx = t.transform.translation.x
            ry = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            self.robot_pose = (rx, ry, yaw)
        except Exception:
            pass

    def world_to_pixel(self, wx: float, wy: float, info) -> tuple:
        """Transforms world coordinate (wx, wy) into canvas pixel (px, py) on left map panel."""
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        mw = info.width
        mh = info.height

        gx = (wx - ox) / res
        gy = (wy - oy) / res

        scale_x = (self.map_w - 40) / max(1, mw)
        scale_y = (self.map_h - 90) / max(1, mh)
        scale = min(scale_x, scale_y)

        disp_w = int(mw * scale)
        disp_h = int(mh * scale)
        offset_x = int((self.map_w - disp_w) / 2)
        offset_y = int(45 + ((self.map_h - 75) - disp_h) / 2)

        px = int(offset_x + gx * scale)
        py = int(offset_y + (mh - gy) * scale)
        return (px, py)

    def render_loop(self):
        self.update_robot_pose()
        canvas = np.zeros((self.total_h, self.total_w, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 28)

        if self.latest_map is None:
            cv2.putText(canvas, "Awaiting SLAM Map from SLAM Toolbox...", (40, self.map_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            cv2.putText(canvas, "Aligning TF frames: map -> odom -> base_footprint", (40, (self.map_h // 2) + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
            self.render_sidebar(canvas)
            self.display_and_publish(canvas)
            return

        grid = self.latest_map
        info = grid.info
        w = info.width
        h = info.height
        raw_data = np.array(grid.data, dtype=np.int8).reshape((h, w))

        # Base map coloring
        map_img = np.zeros((h, w, 3), dtype=np.uint8)
        # Unknown: dark lawn green tint
        map_img[raw_data == -1] = [24, 38, 26]
        # Free space: clean promenade stone
        map_img[raw_data == 0] = [235, 238, 240]
        # Obstacles / boundary hedges: deep charcoal
        map_img[raw_data >= 50] = [25, 30, 25]
        # Mid-cost inflation: subtle slate
        mid_mask = (raw_data > 0) & (raw_data < 50)
        map_img[mid_mask] = [190, 200, 205]

        map_img = cv2.flip(map_img, 0)

        scale_x = (self.map_w - 40) / max(1, w)
        scale_y = (self.map_h - 90) / max(1, h)
        scale = min(scale_x, scale_y)
        disp_w = int(w * scale)
        disp_h = int(h * scale)

        resized_map = cv2.resize(map_img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        offset_x = int((self.map_w - disp_w) / 2)
        offset_y = int(45 + ((self.map_h - 75) - disp_h) / 2)
        canvas[offset_y:offset_y + disp_h, offset_x:offset_x + disp_w] = resized_map
        cv2.rectangle(canvas, (offset_x - 1, offset_y - 1), (offset_x + disp_w + 1, offset_y + disp_h + 1), (60, 65, 75), 1)

        # 1. Draw Trajectory Trail
        if len(self.latest_trajectory) > 1:
            pts = [self.world_to_pixel(x, y, info) for x, y in self.latest_trajectory]
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], (180, 230, 50), 2, cv2.LINE_AA)

        # 2. Draw Planned A* Global Path
        if len(self.latest_path) > 1:
            path_pts = [self.world_to_pixel(x, y, info) for x, y in self.latest_path]
            for i in range(len(path_pts) - 1):
                cv2.line(canvas, path_pts[i], path_pts[i + 1], (0, 210, 255), 2, cv2.LINE_AA)
            for pt in path_pts[::4]:
                cv2.circle(canvas, pt, 2, (0, 180, 255), -1)

        # 3. Draw ONLY Dynamically Discovered Litter Pins (Zero prior knowledge)
        for k, t in self.discovered_trash.items():
            tx, ty = self.world_to_pixel(t['x'], t['y'], info)
            is_collected = t['collected']
            badge_color = (80, 90, 100) if is_collected else t['color']

            cv2.circle(canvas, (tx, ty), 10, badge_color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 7, (20, 22, 28) if not is_collected else (40, 80, 45), -1, cv2.LINE_AA)

            badge_sym = "v" if is_collected else "!"
            cv2.putText(canvas, badge_sym, (tx - 4, ty + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Draw Robot Current Pose & Heading Arrow [R]
        if self.robot_pose is not None:
            rx, ry, ryaw = self.robot_pose
            rpx, rpy = self.world_to_pixel(rx, ry, info)

            arrow_len = 18
            ax = int(rpx + arrow_len * math.cos(ryaw))
            ay = int(rpy - arrow_len * math.sin(ryaw))

            cv2.circle(canvas, (rpx, rpy), 12, (255, 140, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (rpx, rpy), 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (rpx, rpy), (ax, ay), (0, 255, 255), 2, tipLength=0.35)
            cv2.putText(canvas, "R", (rpx - 4, rpy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Top Banner on Left Panel
        cv2.rectangle(canvas, (0, 0), (self.map_w, 40), (16, 18, 22), -1)
        cv2.putText(canvas, "AUTONOMOUS GARDEN CLEANER - 2D SLAM MAP", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 240, 120), 2, cv2.LINE_AA)

        # Render Right Sidebar
        self.render_sidebar(canvas)
        self.display_and_publish(canvas)

    def render_sidebar(self, canvas: np.ndarray):
        sx = self.map_w
        cv2.line(canvas, (sx, 0), (sx, self.total_h), (50, 55, 65), 1)
        cv2.rectangle(canvas, (sx + 1, 0), (self.total_w, self.total_h), (22, 25, 32), -1)

        # Panel Title
        cv2.putText(canvas, "CLEANER TELEMETRY HUD", (sx + 16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 240, 120), 2, cv2.LINE_AA)
        cv2.line(canvas, (sx + 16, 34), (self.total_w - 16, 34), (45, 50, 60), 1)

        y = 54
        # Section 1: Robot Pose & Motion
        if self.robot_pose is not None:
            rx, ry, ryaw = self.robot_pose
            cv2.putText(canvas, f"Pose: ({rx:.2f}, {ry:.2f})m", (sx + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 225, 230), 1, cv2.LINE_AA)
            y += 20
            yaw_deg = math.degrees(ryaw)
            cv2.putText(canvas, f"Heading: {yaw_deg:.1f} deg", (sx + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 185, 195), 1, cv2.LINE_AA)
            y += 20
            cv2.putText(canvas, f"Speed: {self.current_cmd_v:.2f} m/s", (sx + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 185, 195), 1, cv2.LINE_AA)
            y += 22

        # State badge
        clean_state = self.status_text
        for emoji in ["🗺️", "🎯", "🧹", "🎉", "✓", "⚠️", "🚶", "🌿"]:
            clean_state = clean_state.replace(emoji, "")
        clean_state = clean_state.strip()[:28]
        cv2.rectangle(canvas, (sx + 16, y - 2), (self.total_w - 16, y + 22), (32, 36, 46), -1)
        cv2.rectangle(canvas, (sx + 16, y - 2), (self.total_w - 16, y + 22), (0, 200, 220), 1)
        cv2.putText(canvas, clean_state, (sx + 22, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1, cv2.LINE_AA)
        y += 38

        # Section 2: Score & Hopper Bin Telemetry
        cv2.line(canvas, (sx + 16, y), (self.total_w - 16, y), (45, 50, 60), 1)
        y += 18
        cv2.putText(canvas, "CLEANING SCORE & HOPPER", (sx + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 0), 2, cv2.LINE_AA)
        y += 22

        collected_count = sum(1 for item in self.discovered_trash.values() if item['collected'])
        cv2.putText(canvas, f"Score: {self.score} PTS  [Hopper: {self.bin_capacity_pct:.0f}%]", (sx + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 230, 80), 1, cv2.LINE_AA)
        y += 22

        # Hopper fill bar
        bar_w = self.total_w - sx - 32
        fill_pct = max(0.0, min(100.0, self.bin_capacity_pct))
        fill_w = int(bar_w * (fill_pct / 100.0))
        cv2.rectangle(canvas, (sx + 16, y), (sx + 16 + bar_w, y + 10), (40, 45, 55), -1)
        cv2.rectangle(canvas, (sx + 16, y), (sx + 16 + fill_w, y + 10), (50, 220, 80), -1)
        cv2.rectangle(canvas, (sx + 16, y), (sx + 16 + bar_w, y + 10), (80, 90, 100), 1)
        y += 26

        # Section 3: GPS Telemetry
        cv2.line(canvas, (sx + 16, y), (self.total_w - 16, y), (45, 50, 60), 1)
        y += 18
        cv2.putText(canvas, "GNSS / GPS TELEMETRY", (sx + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 2, cv2.LINE_AA)
        y += 20
        gps_display = self.gps_text
        for emoji in ["🛰️", "📍", "✓", "°"]:
            gps_display = gps_display.replace(emoji, "")
        gps_display = gps_display.replace("GPS: ", "").strip()
        for line in gps_display.split("|"):
            cv2.putText(canvas, line.strip(), (sx + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 210, 230), 1, cv2.LINE_AA)
            y += 18
        y += 8

        # Section 4: AI Vision Litter Discovery Feed (Zero prior knowledge)
        cv2.line(canvas, (sx + 16, y), (self.total_w - 16, y), (45, 50, 60), 1)
        y += 18
        cv2.putText(canvas, "AI VISION DISCOVERIES", (sx + 16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 120, 80), 2, cv2.LINE_AA)
        y += 22

        if not self.discovered_trash:
            cv2.putText(canvas, "[ Scanning park for litter... ]", (sx + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 140, 160), 1, cv2.LINE_AA)
            y += 18
            cv2.putText(canvas, "  Frontier Exploration Active", (sx + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 180, 100), 1, cv2.LINE_AA)
            y += 22
        else:
            rx = self.robot_pose[0] if self.robot_pose is not None else 0.0
            ry = self.robot_pose[1] if self.robot_pose is not None else 0.0

            for k, t in list(self.discovered_trash.items())[:8]:
                dist_t = math.hypot(t['x'] - rx, t['y'] - ry)
                collected = t['collected']
                badge_sym = "[v]" if collected else "[!]"
                color = (80, 220, 80) if collected else t['color']
                status_str = "IN HOPPER" if collected else f"{dist_t:.1f}m away"

                cv2.putText(canvas, f"{badge_sym} {t['name']}", (sx + 16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                y += 16
                cv2.putText(canvas, f"    ({t['x']:.1f}, {t['y']:.1f}) | {status_str}", (sx + 16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.31, (130, 140, 150), 1, cv2.LINE_AA)
                y += 20

    def display_and_publish(self, canvas: np.ndarray):
        if self.gui_available:
            cv2.imshow('Garden Cleaner - Live SLAM Map & Mission HUD', canvas)
            cv2.waitKey(1)

        try:
            msg = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
            self.map_image_pub.publish(msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SlamMapVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.gui_available:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
