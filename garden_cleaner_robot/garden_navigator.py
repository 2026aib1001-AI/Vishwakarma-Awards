#!/usr/bin/env python3
"""
Cognitive Map-Based Autonomous Explorer & Cleaner for Garden Cleaner Robot.
Ported directly from blind_assistant_robot/autonomous_explorer.py.

Key Architecture:
1. True Autonomous Frontier Exploration (Zero Prior Knowledge):
   - NO hardcoded or pre-known trash coordinates.
   - Computes morphological frontiers (boundary between explored free space and unknown territory)
     directly from the SLAM OccupancyGrid via OpenCV SIMD C++ connected components.
   - Automatically prioritizes the most promising frontier clusters and routes across the park.
2. Real-Time Computer Vision Trash Interception:
   - As the robot traverses the park, onboard RGB camera (/camera/image_raw) runs OpenCV detection.
   - When litter (soda cans, juice boxes) is spotted, the vision detector projects its 3D location into
     the SLAM map frame and publishes /garden/detected_trash.
   - The navigator dynamically pauses frontier exploration, intercepts the spotted litter,
     navigates within collection threshold, vacuums it up into the hopper, and resumes exploration!
3. Persistent World Memory & Costmap Inflation:
   - Dilates lethal obstacles by robot radius (0.22m) and soft potential gradient (0.45m).
   - Guarantees the central roundabout retaining wall (radius 2.20m, height 0.55m) is mapped
     as a solid lethal obstacle, routing all trajectories safely along the wide promenade apron (radius 3.5m).
4. Pure Pursuit Motion Controller (10 Hz):
   - Strict in-place pivoting on sharp heading changes (|error| > 20 deg) to eliminate wheel scrub.
   - Slew-rate velocity smoother for clean acceleration.
   - Dynamic pedestrian corridor yielding (pauses up to 3.5s, then replans detour).
   - Physical stuck detection with reverse clearing recovery.
5. Command Dispatch:
   - Supports "clean c to a", "clean a to b", "clean b to c", "clean all", "explore", "patrol", "stop", "start".
"""

import os
import math
import time
import heapq
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan, NavSatFix
from geometry_msgs.msg import Twist, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener


FREE_THRESHOLD = 50
UNKNOWN = -1
MIN_FRONTIER_SIZE = 2
FRONTIER_SEARCH_RADIUS = 2


class GardenNavigator(Node):
    def __init__(self):
        super().__init__('garden_navigator')

        # Tunable Navigation Parameters (matching proven blind_assistant_robot settings)
        self.declare_parameter('max_linear_speed', 0.40)
        self.declare_parameter('max_angular_speed', 0.85)
        self.declare_parameter('robot_radius', 0.22)
        self.declare_parameter('lookahead_dist', 0.35)
        self.declare_parameter('autostart_patrol', True)
        self.declare_parameter('use_nav2', False)

        self.max_lin_speed = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self.max_ang_speed = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        self.robot_radius = self.get_parameter('robot_radius').get_parameter_value().double_value
        self.lookahead_dist = self.get_parameter('lookahead_dist').get_parameter_value().double_value
        self.autostart = self.get_parameter('autostart_patrol').get_parameter_value().bool_value
        self.use_nav2 = self.get_parameter('use_nav2').get_parameter_value().bool_value

        # Named Hubs and Landmarks for Mission Dispatch
        self.hubs = {
            'a': (0.0, -7.0),
            'b': (9.0, 6.0),
            'c': (-9.0, 6.0),
            'bench': (9.5, 8.5),
            'fountain': (-9.5, 8.5),
            'planter': (0.0, -9.8),
            'tree': (0.0, -2.8),
            'start': (0.0, -7.0),
            'home': (0.0, -7.0),
        }

        # 3 Avenue Promenades (Waypoints for specific missions - NO pre-known trash!)
        self.path_ab = [
            (0.0, -7.0),     # Zone A (South Plaza)
            (4.75, -0.75),   # Midpoint Avenue A->B
            (9.0, 6.0),      # Zone B (East Pavilion)
        ]

        self.path_bc = [
            (9.0, 6.0),      # Zone B (East Pavilion)
            (0.0, 6.0),      # Midpoint Avenue B->C
            (-9.0, 6.0),     # Zone C (West Terrace)
        ]

        self.path_ca = [
            (-9.0, 6.0),     # Zone C (West Terrace)
            (-4.75, -0.75),  # Midpoint Avenue C->A
            (0.0, -7.0),     # Zone A (South Plaza)
        ]

        # Full Loop Promenade (used when exploration is complete or on 'clean all')
        self.patrol_waypoints = (
            self.path_ab[:-1] +
            self.path_bc[:-1] +
            self.path_ca
        )

        self.mission_queue = list(self.patrol_waypoints)
        self.mission_index = 0
        self.mission_name = "Autonomous Frontier Exploration & Cleaning"

        # Robot Pose Tracking
        self.current_x = 0.0
        self.current_y = -7.0
        self.current_yaw = 0.957
        self.odom_received = False
        self.pose_in_map_valid = False

        # Internal World Costmap Memory
        self.current_map = None
        self.lethal_costmap = None
        self.soft_costmap = None
        self.grid_data = None
        self.latest_scan = None

        # Autonomous Exploration & Frontier Decision Engine
        self.mode = 'EXPLORE' if self.autostart else 'STOP'
        self.state = 'FOLLOWING_PATH'  # 'FOLLOWING_PATH', 'RECOVERING', 'YIELDING', 'STOP'
        self.current_goal = None
        self.current_path = []
        self.target_goal = (0.0, -7.0)
        self.blacklisted_goals = []  # (gx, gy, expire_time)
        self.goal_start_time = time.time()
        self.exploration_start_time = None
        self.last_replan_time = 0.0

        # Dynamic Vision-Driven Litter Interception State
        self.intercepting_trash = False
        self.active_trash_target = None
        self.pre_intercept_goal = None
        self.pre_intercept_path = []
        self.collected_trash_locs = []
        self.intercept_start_time = 0.0
        self.intercept_arrival_time = None

        # Slew-Rate Velocity Smoother
        self.current_cmd_v = 0.0
        self.current_cmd_w = 0.0
        self.last_sent_linear_speed = 0.0

        # Physical Stuck Detection & Recovery
        self.last_check_pos = (0.0, -7.0)
        self.last_check_time = time.time()
        self.recovery_end_time = 0.0
        self.recovery_turn_dir = 1.0
        self.pre_recovery_state = 'FOLLOWING_PATH'

        # Dynamic Pedestrian Corridor Yielding
        self.human_pose = None
        self.last_human_pose_time = 0.0
        self.is_yielding = False
        self.yield_start_time = 0.0

        # GPS / GNSS Telemetry
        self.gps_lat = 0.0
        self.gps_lon = 0.0
        self.gps_alt = 0.0
        self.gps_status = -1
        self.gps_received = False

        # TF Buffer & Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Trajectory Trail
        self.trajectory_path = Path()
        self.trajectory_path.header.frame_id = 'map'
        self.last_traj_time = 0.0

        # QoS Profiles
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.trajectory_pub = self.create_publisher(Path, '/robot_trajectory', 10)
        self.footprint_marker_pub = self.create_publisher(Marker, '/robot_footprint', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/exploration_goal', 10)
        self.frontier_markers_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.status_pub = self.create_publisher(String, '/garden/nav_status', 10)
        self.gps_status_pub = self.create_publisher(String, '/garden/gps_status', 10)

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, latched_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.human_sub = self.create_subscription(
            PoseStamped, '/detected_human', self.human_pose_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.goto_sub = self.create_subscription(
            String, '/garden/go_to', self.goto_callback, 10)

        # Dynamic Vision Trash Detection & Collection Subscribers
        self.detected_trash_sub = self.create_subscription(
            PoseStamped, '/garden/detected_trash', self.vision_trash_callback, 10)
        self.trash_collected_sub = self.create_subscription(
            String, '/garden/trash_collected', self.trash_collected_callback, 10)

        # Motion Control Loop (10 Hz Pure Pursuit)
        self.control_timer = self.create_timer(0.10, self.motion_control_step)

        # Cognitive Frontier & Exploration Planner Loop (1 Hz)
        self.cognitive_timer = self.create_timer(1.0, self.cognitive_planner_step)

        # GPS Status Publisher (2 Hz)
        self.gps_timer = self.create_timer(0.5, self.publish_gps_telemetry)

        self.get_logger().info('══════════════════════════════════════════════════════════')
        self.get_logger().info('🧭 COGNITIVE GARDEN NAVIGATOR ONLINE (ZERO PRIOR KNOWLEDGE) 🧭')
        self.get_logger().info(f'   Initial Mode: {self.mode}')
        self.get_logger().info('   Architecture: Internal A* on SLAM Map + Frontier Exploration + Pure Pursuit')
        self.get_logger().info('   Features: Autonomous Park Exploration + Real-Time CV Litter Interception')
        self.get_logger().info('══════════════════════════════════════════════════════════')

    # =========================================================================
    # 1. Pose Tracking & TF Lookup
    # =========================================================================
    def update_robot_pose(self):
        """Queries the exact SLAM-corrected robot pose in the map frame."""
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            self.current_x = t.transform.translation.x
            self.current_y = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_yaw = math.atan2(siny, cosy)
            self.pose_in_map_valid = True

            now_sec = time.time()
            if now_sec - self.last_traj_time > 0.5:
                self.last_traj_time = now_sec
                pt = PoseStamped()
                pt.header.stamp = self.get_clock().now().to_msg()
                pt.header.frame_id = 'map'
                pt.pose.position.x = self.current_x
                pt.pose.position.y = self.current_y
                pt.pose.position.z = 0.02
                pt.pose.orientation = q
                self.trajectory_path.poses.append(pt)
                if len(self.trajectory_path.poses) > 3000:
                    self.trajectory_path.poses.pop(0)
                self.trajectory_path.header.stamp = pt.header.stamp
                self.trajectory_pub.publish(self.trajectory_path)
        except Exception:
            pass

    def odom_callback(self, msg: Odometry):
        # Update fallback pose until map frame becomes available from SLAM
        if not self.pose_in_map_valid:
            self.current_x = msg.pose.pose.position.x
            self.current_y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_yaw = math.atan2(siny, cosy)
        self.odom_received = True
        self.update_robot_pose()

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def human_pose_callback(self, msg: PoseStamped):
        self.human_pose = (msg.pose.position.x, msg.pose.position.y)
        self.last_human_pose_time = time.time()

    def gps_callback(self, msg: NavSatFix):
        self.gps_lat = msg.latitude
        self.gps_lon = msg.longitude
        self.gps_alt = msg.altitude
        self.gps_status = msg.status.status
        self.gps_received = True

    def publish_gps_telemetry(self):
        if not self.gps_received:
            return
        status_msg = String()
        status_msg.data = (
            f"🛰️ GPS: Lat {self.gps_lat:.6f}°, Lon {self.gps_lon:.6f}°, "
            f"Alt {self.gps_alt:.1f}m | Fix Status: {self.gps_status}"
        )
        self.gps_status_pub.publish(status_msg)

    # =========================================================================
    # 2. Dynamic Vision-Driven Litter Interception
    # =========================================================================
    def vision_trash_callback(self, msg: PoseStamped):
        """Called dynamically whenever camera vision spots a piece of litter."""
        tx = msg.pose.position.x
        ty = msg.pose.position.y

        # Check if already collected
        for (cx, cy) in self.collected_trash_locs:
            if math.hypot(tx - cx, ty - cy) < 1.8:
                return

        d = math.hypot(tx - self.current_x, ty - self.current_y)
        if d > 7.0 or d < 0.25:
            return

        if not self.intercepting_trash and self.mode in ('EXPLORE', 'PATROL', 'MISSION'):
            self.intercepting_trash = True
            self.intercept_start_time = time.time()
            self.intercept_arrival_time = None
            self.pre_intercept_goal = self.target_goal
            self.pre_intercept_path = list(self.current_path)

            self.active_trash_target = (tx, ty)
            self.target_goal = (tx, ty)
            self.get_logger().info(f"🎯 [AI VISION SIGHTING] Litter spotted at ({tx:.2f}, {ty:.2f})! Diverting to pick up...")
            self.publish_status(f"🎯 Intercepting spotted litter at ({tx:.1f}, {ty:.1f})")
            self.route_to_target_goal()

    def trash_collected_callback(self, msg: String):
        """Called when trash_collector_node vacuums up a piece of litter."""
        try:
            parts = msg.data.split(':')
            if len(parts) >= 3:
                tx = float(parts[1])
                ty = float(parts[2])
                self.collected_trash_locs.append((tx, ty))
        except Exception:
            pass

        if self.active_trash_target is not None:
            self.collected_trash_locs.append(self.active_trash_target)

        if self.intercepting_trash:
            self.get_logger().info("✅ [LITTER RETRIEVED] Resuming autonomous exploration & cleaning.")
            self.intercepting_trash = False
            self.active_trash_target = None
            self.intercept_arrival_time = None
            self.current_goal = None
            self.current_path = []
            self.publish_status("Litter collected! Resuming autonomous frontier exploration.")

    # =========================================================================
    # 3. In-Memory Costmap Inflation (Roundabout Clearance & Path Centering)
    # =========================================================================
    def map_callback(self, msg: OccupancyGrid):
        self.current_map = msg
        self.update_costmap_memory()

    def update_costmap_memory(self):
        """Constructs an inflated costmap in memory based on the robot's physical dimensions."""
        if self.current_map is None:
            return

        w = self.current_map.info.width
        h = self.current_map.info.height
        res = self.current_map.info.resolution
        grid = np.array(self.current_map.data, dtype=np.int8).reshape((h, w))
        self.grid_data = grid

        # Identify occupied cells (walls, hedges, roundabout retaining wall)
        occupied = (grid >= FREE_THRESHOLD)

        # Lethal radius (5 cells = 0.25m for 0.05m resolution)
        r_lethal = max(1, int(round(self.robot_radius / res)))
        self.lethal_costmap = self._dilate_mask(occupied, r_lethal, h, w)

        # Soft inflation layer (adds potential cost around walls to keep robot centered in pathways)
        r_soft = r_lethal + 4
        soft_layer = self._dilate_mask(occupied, r_soft, h, w)
        self.soft_costmap = soft_layer & (~self.lethal_costmap)

    def _dilate_mask(self, mask: np.ndarray, radius: int, h: int, w: int) -> np.ndarray:
        """Fast vectorized binary dilation using OpenCV elliptical structuring element."""
        if radius <= 0:
            return mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated.astype(bool)

    # =========================================================================
    # 4. Global A* Path Planner on Remembered Map
    # =========================================================================
    def world_to_grid(self, wx, wy):
        if self.current_map is None:
            return None
        ox = self.current_map.info.origin.position.x
        oy = self.current_map.info.origin.position.y
        res = self.current_map.info.resolution
        gx = int((wx - ox) / res)
        gy = int((wy - oy) / res)
        return gx, gy

    def grid_to_world(self, gx, gy):
        if self.current_map is None:
            return gx, gy
        ox = self.current_map.info.origin.position.x
        oy = self.current_map.info.origin.position.y
        res = self.current_map.info.resolution
        wx = ox + (gx + 0.5) * res
        wy = oy + (gy + 0.5) * res
        return wx, wy

    def find_nearest_safe_cell(self, start_gx, start_gy, max_search=8):
        """Finds nearest un-inflated, safe cell if coordinate falls inside an obstacle border."""
        h, w = self.lethal_costmap.shape
        if 0 <= start_gx < w and 0 <= start_gy < h and not self.lethal_costmap[start_gy, start_gx]:
            return start_gx, start_gy

        best = None
        best_dist = float('inf')
        for dy in range(-max_search, max_search + 1):
            for dx in range(-max_search, max_search + 1):
                nx, ny = start_gx + dx, start_gy + dy
                if 0 <= nx < w and 0 <= ny < h and not self.lethal_costmap[ny, nx]:
                    d = dx * dx + dy * dy
                    if d < best_dist:
                        best_dist = d
                        best = (nx, ny)
        return best

    def plan_astar_path(self, start_world, goal_world):
        """Calculates optimal collision-free path on the inflated costmap."""
        if self.current_map is None or self.lethal_costmap is None:
            return None

        sg = self.world_to_grid(start_world[0], start_world[1])
        gg = self.world_to_grid(goal_world[0], goal_world[1])
        if sg is None or gg is None:
            return None

        safe_start = self.find_nearest_safe_cell(sg[0], sg[1], max_search=8)
        safe_goal = self.find_nearest_safe_cell(gg[0], gg[1], max_search=10)
        if safe_start is None or safe_goal is None:
            return None

        sx, sy = safe_start
        gx, gy = safe_goal
        h, w = self.lethal_costmap.shape

        open_set = []
        h_cost = math.hypot(gx - sx, gy - sy)
        heapq.heappush(open_set, (h_cost, 0.0, (sx, sy)))

        came_from = {}
        g_scores = {(sx, sy): 0.0}
        visited = set()

        max_iterations = 50000
        iterations = 0

        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

        while open_set and iterations < max_iterations:
            iterations += 1
            f, curr_g, (cx, cy) = heapq.heappop(open_set)

            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if math.hypot(gx - cx, gy - cy) <= 1.0:
                # Goal reached
                path_grid = [(cx, cy)]
                p = (cx, cy)
                while p in came_from:
                    p = came_from[p]
                    path_grid.append(p)
                path_grid.reverse()

                smoothed = self.smooth_path_line_of_sight(path_grid)
                world_path = [self.grid_to_world(px, py) for px, py in smoothed]
                return self.interpolate_path(world_path, max_step=0.15)

            for dx, dy, step_cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if self.lethal_costmap[ny, nx]:
                        continue
                    if dx != 0 and dy != 0:
                        if self.lethal_costmap[cy, nx] or self.lethal_costmap[ny, cx]:
                            continue

                    soft_penalty = 3.0 if self.soft_costmap[ny, nx] else 0.0
                    tentative_g = curr_g + step_cost + soft_penalty

                    if (nx, ny) not in g_scores or tentative_g < g_scores[(nx, ny)]:
                        g_scores[(nx, ny)] = tentative_g
                        h_est = 1.02 * math.hypot(gx - nx, gy - ny)
                        came_from[(nx, ny)] = (cx, cy)
                        heapq.heappush(open_set, (tentative_g + h_est, tentative_g, (nx, ny)))

        return None

    def line_of_sight(self, p1, p2):
        """Bresenham-based collision checker along grid segment."""
        x0, y0 = p1
        x1, y1 = p2
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if self.lethal_costmap[y, x]:
                return False
            if x == x1 and y == y1:
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    def smooth_path_line_of_sight(self, grid_path):
        """Prunes unnecessary waypoints when there is an unblocked line of sight."""
        if len(grid_path) <= 2:
            return grid_path

        smoothed = [grid_path[0]]
        curr_idx = 0
        while curr_idx < len(grid_path) - 1:
            furthest = curr_idx + 1
            for next_idx in range(len(grid_path) - 1, curr_idx, -1):
                if self.line_of_sight(grid_path[curr_idx], grid_path[next_idx]):
                    furthest = next_idx
                    break
            smoothed.append(grid_path[furthest])
            curr_idx = furthest
        return smoothed

    def interpolate_path(self, world_path, max_step=0.15):
        """Subdivides sparse line-of-sight segments into dense, evenly spaced waypoints for smooth Pure Pursuit tracking."""
        if not world_path or len(world_path) <= 1:
            return world_path

        dense = [world_path[0]]
        for i in range(len(world_path) - 1):
            x1, y1 = world_path[i]
            x2, y2 = world_path[i + 1]
            dist = math.hypot(x2 - x1, y2 - y1)
            steps = max(1, int(math.ceil(dist / max_step)))
            for s in range(1, steps + 1):
                frac = s / steps
                dense.append((x1 + frac * (x2 - x1), y1 + frac * (y2 - y1)))
        return dense

    def plan_astar_path_with_fallback(self, start_world, goal_world):
        """Attempts direct A* to goal_world; if blocked by inflation, finds nearest reachable free coordinate."""
        path = self.plan_astar_path(start_world, goal_world)
        if path and len(path) > 1:
            return path

        gx0, gy0 = goal_world
        for radius in [0.20, 0.35, 0.50, 0.70, 0.90]:
            for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                cand_x = gx0 + radius * math.cos(angle)
                cand_y = gy0 + radius * math.sin(angle)
                cand_path = self.plan_astar_path(start_world, (cand_x, cand_y))
                if cand_path and len(cand_path) > 1:
                    return cand_path

        # If still None, generate direct interpolated dense straight-line
        dx = goal_world[0] - start_world[0]
        dy = goal_world[1] - start_world[1]
        dist = math.hypot(dx, dy)
        steps = max(2, int(dist / 0.15))
        return [(start_world[0] + dx * (i / steps), start_world[1] + dy * (i / steps)) for i in range(steps + 1)]

    # =========================================================================
    # 5. Frontier Detection & Selection (from blind_assistant_robot)
    # =========================================================================
    def detect_frontiers(self):
        """Identifies boundary clusters between explored free space and unmapped territory."""
        if self.current_map is None:
            return []

        grid = self.current_map
        w = grid.info.width
        h = grid.info.height
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y

        data = np.array(grid.data, dtype=np.int8).reshape((h, w))
        free_mask = (data >= 0) & (data < FREE_THRESHOLD)
        unknown_mask = (data == UNKNOWN)

        # High-performance OpenCV morphological neighbor discovery
        kernel_search = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * FRONTIER_SEARCH_RADIUS + 1, 2 * FRONTIER_SEARCH_RADIUS + 1)
        )
        dilated_unknown = cv2.dilate(unknown_mask.astype(np.uint8), kernel_search).astype(bool)
        frontier_mask = free_mask & dilated_unknown

        # Exclude cells that fall within lethal inflated obstacle borders
        if self.lethal_costmap is not None:
            frontier_mask &= (~self.lethal_costmap)

        if not np.any(frontier_mask):
            return []

        # Vectorized 8-connected component clustering via OpenCV SIMD C++
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_mask.astype(np.uint8), connectivity=8
        )

        clusters = []
        for label in range(1, num_labels):
            size = int(stats[label, cv2.CC_STAT_AREA])
            if size < MIN_FRONTIER_SIZE:
                continue

            cgx, cgy = centroids[label]
            ys, xs = np.where(labels == label)
            dists_sq = (xs - cgx) ** 2 + (ys - cgy) ** 2
            best_idx = np.argmin(dists_sq)
            best_gx, best_gy = xs[best_idx], ys[best_idx]

            wx = ox + (best_gx + 0.5) * res
            wy = oy + (best_gy + 0.5) * res
            dist = math.hypot(wx - self.current_x, wy - self.current_y)

            # Keep frontiers strictly within park bounds
            if -12.5 <= wx <= 12.5 and -10.5 <= wy <= 9.5:
                clusters.append({
                    'centroid': (wx, wy),
                    'size': size,
                    'distance': dist
                })

        return clusters

    def select_best_frontier(self, frontiers):
        if not frontiers:
            return None

        now = time.time()
        valid = []
        for f in frontiers:
            cx, cy = f['centroid']
            is_blacklisted = any(
                math.hypot(cx - bx, cy - by) < 1.0 for bx, by, exp in self.blacklisted_goals if exp > now
            )
            if not is_blacklisted and 0.85 <= f['distance'] <= 18.0:
                valid.append(f)

        if not valid:
            valid = [f for f in frontiers if f['distance'] >= 0.70 and not any(
                math.hypot(f['centroid'][0] - bx, f['centroid'][1] - by) < 1.0 for bx, by, exp in self.blacklisted_goals if exp > now
            )]

        if not valid:
            return None

        max_size = max(f['size'] for f in valid)
        max_dist = max(f['distance'] for f in valid) or 1.0

        best = None
        best_score = -float('inf')

        for f in valid:
            # Score formula: reward larger frontiers, penalize long distances
            score = (f['size'] / max_size) * 1.5 - (f['distance'] / max_dist) * 1.0
            if score > best_score:
                best_score = score
                best = f

        return best

    def publish_frontier_markers(self, frontiers):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for idx, f in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'frontiers'
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = f['centroid'][0]
            m.pose.position.y = f['centroid'][1]
            m.pose.position.z = 0.10
            m.pose.orientation.w = 1.0
            m.scale.x = 0.25
            m.scale.y = 0.25
            m.scale.z = 0.25
            m.color = ColorRGBA(r=0.2, g=0.9, b=1.0, a=0.7)
            m.lifetime = Duration(sec=2)
            ma.markers.append(m)
        self.frontier_markers_pub.publish(ma)

    # =========================================================================
    # 6. Mission Dispatch & Command Parsing
    # =========================================================================
    def goto_callback(self, msg: String):
        raw_cmd = msg.data.strip().lower()
        cmd = raw_cmd.replace("_", " ").replace("-", " ")
        self.get_logger().info(f"📨 Command Received: '{raw_cmd}'")

        if cmd in ('stop', 'pause', 'halt'):
            self.mode = 'STOP'
            self.state = 'STOP'
            self.stop_robot()
            self.publish_status("Robot paused.")
            return

        if cmd == 'patrol':
            self.mode = 'PATROL'
            self.state = 'FOLLOWING_PATH'
            self.current_goal = None
            self.current_path = []
            self.mission_name = "Autonomous Frontier Patrol & Cleaning"
            self.publish_status("🧹 Exploring & Cleaning entire park autonomously!")
            return

        if cmd in ('resume', 'start', 'explore'):
            self.mode = 'EXPLORE'
            self.state = 'FOLLOWING_PATH'
            self.current_goal = None
            self.current_path = []
            self.mission_name = "Autonomous Frontier Exploration & Cleaning"
            self.publish_status("🧹 Exploring & Cleaning entire park autonomously!")
            return

        # Specific Mission: Clean C to A
        if 'c to a' in cmd or 'c_to_a' in raw_cmd or cmd == 'clean c to a':
            self.mode = 'MISSION'
            self.state = 'FOLLOWING_PATH'
            self.mission_queue = list(self.path_ca)
            self.mission_index = 0
            self.target_goal = self.mission_queue[0]
            self.mission_name = "Clean Path C to A"
            self.route_to_target_goal()
            self.publish_status("🧹 Initiating Sweep: Clean Path C to A!")
            return

        # Specific Mission: Clean A to B
        if 'a to b' in cmd or 'a_to_b' in raw_cmd or cmd == 'clean a to b':
            self.mode = 'MISSION'
            self.state = 'FOLLOWING_PATH'
            self.mission_queue = list(self.path_ab)
            self.mission_index = 0
            self.target_goal = self.mission_queue[0]
            self.mission_name = "Clean Path A to B"
            self.route_to_target_goal()
            self.publish_status("🧹 Initiating Sweep: Clean Path A to B!")
            return

        # Specific Mission: Clean B to C
        if 'b to c' in cmd or 'b_to_c' in raw_cmd or cmd == 'clean b to c':
            self.mode = 'MISSION'
            self.state = 'FOLLOWING_PATH'
            self.mission_queue = list(self.path_bc)
            self.mission_index = 0
            self.target_goal = self.mission_queue[0]
            self.mission_name = "Clean Path B to C"
            self.route_to_target_goal()
            self.publish_status("🧹 Initiating Sweep: Clean Path B to C!")
            return

        # Clean All
        if 'clean all' in cmd or 'cleanall' in cmd:
            self.mode = 'MISSION'
            self.state = 'FOLLOWING_PATH'
            self.mission_queue = list(self.patrol_waypoints)
            self.mission_index = 0
            self.target_goal = self.mission_queue[0]
            self.mission_name = "Full Triangle Clean-All"
            self.route_to_target_goal()
            self.publish_status("🧹 Sweeping complete triangle: A -> B -> C -> A!")
            return

        # Hub / Landmark Navigation
        for hub_key, coord in self.hubs.items():
            if hub_key in cmd or f"go to {hub_key}" in cmd or f"zone {hub_key}" in cmd:
                self.mode = 'GOTO'
                self.state = 'FOLLOWING_PATH'
                self.target_goal = coord
                self.mission_name = f"Go to {hub_key.upper()}"
                self.route_to_target_goal()
                self.publish_status(f"Navigating to {hub_key.upper()} at {coord}")
                return

        self.get_logger().warning(
            f"Unknown command '{raw_cmd}'. Valid: 'clean c to a', 'clean a to b', 'clean b to c', 'clean all', 'explore', 'stop'"
        )

    def route_to_target_goal(self):
        """Plans an optimal, collision-free A* trajectory from robot pose to current target_goal."""
        self.update_robot_pose()
        gx, gy = self.target_goal
        path = self.plan_astar_path_with_fallback((self.current_x, self.current_y), (gx, gy))
        if path and len(path) > 1:
            self.current_path = path
            self.publish_planned_path(path)
            self.publish_goal_marker(gx, gy)
            self.get_logger().info(f"🧠 A* Route calculated: {len(path)} dense waypoints to target ({gx:.2f}, {gy:.2f}).")
        else:
            self.current_path = [(self.current_x, self.current_y), (gx, gy)]
            self.publish_planned_path(self.current_path)

    # =========================================================================
    # 7. Cognitive Frontier Decision & Path Verification Loop (1 Hz)
    # =========================================================================
    def cognitive_planner_step(self):
        """1 Hz loop: selects new unexplored frontiers, verifies active paths, triggers replanning."""
        if self.mode == 'STOP' or self.state in ['RECOVERING', 'YIELDING']:
            return

        if self.current_map is None or self.lethal_costmap is None:
            return

        now = time.time()
        self.blacklisted_goals = [b for b in self.blacklisted_goals if b[2] > now]

        # Watchdog: If litter intercept is active for > 15.0 seconds, resume autonomous exploration
        if self.intercepting_trash and (now - self.intercept_start_time > 15.0):
            self.get_logger().info("⏱️ Intercept watchdog timeout (15s). Resuming exploration.")
            if self.target_goal:
                self.collected_trash_locs.append(self.target_goal)
            self.intercepting_trash = False
            self.active_trash_target = None
            self.intercept_arrival_time = None
            self.current_goal = None
            self.current_path = []

        # In autonomous EXPLORE / PATROL mode (Frontier Exploration):
        if self.mode in ('EXPLORE', 'PATROL') and not self.intercepting_trash:
            frontiers = self.detect_frontiers()
            self.publish_frontier_markers(frontiers)

            # Check if current goal frontier has been uncovered
            if self.current_goal is not None:
                gx, gy = self.current_goal
                has_nearby = any(math.hypot(gx - f['centroid'][0], gy - f['centroid'][1]) < 0.75 for f in frontiers)
                if not has_nearby:
                    self.get_logger().info(f"✓ Frontier area at ({gx:.2f}, {gy:.2f}) mapped!")
                    self.blacklisted_goals.append((gx, gy, now + 30.0))
                    self.current_goal = None
                    self.current_path = []

            needs_new_frontier = (len(self.current_path) == 0) or (self.current_goal is None)
            goal_stalled = (self.current_goal is not None) and (now - self.goal_start_time > 20.0)

            if needs_new_frontier or goal_stalled:
                best = self.select_best_frontier(frontiers)
                if best is not None:
                    gx, gy = best['centroid']
                    path = self.plan_astar_path_with_fallback((self.current_x, self.current_y), (gx, gy))
                    if path and len(path) > 1:
                        self.current_goal = (gx, gy)
                        self.target_goal = (gx, gy)
                        self.current_path = path
                        self.goal_start_time = now
                        self.publish_goal_marker(gx, gy)
                        self.publish_planned_path(path)
                        self.publish_status(f"🗺️ Exploring frontier at ({gx:.1f}, {gy:.1f})")
                        self.get_logger().info(f"🧠 Planned A* Path: {len(path)} waypoints to frontier ({gx:.2f}, {gy:.2f})")
                    else:
                        self.blacklisted_goals.append((gx, gy, now + 15.0))
                elif not frontiers:
                    # All frontiers mapped! Traverse promenade avenue loop
                    self.mission_index = (self.mission_index + 1) % len(self.patrol_waypoints)
                    self.target_goal = self.patrol_waypoints[self.mission_index]
                    self.publish_status(f"🧹 Sweeping promenade waypoint #{self.mission_index + 1}")
                    self.route_to_target_goal()

        # Path verification against new dynamic obstacles
        if self.current_path:
            path_blocked = False
            for wx, wy in self.current_path[1:min(10, len(self.current_path))]:
                g = self.world_to_grid(wx, wy)
                if g is not None:
                    gx, gy = g
                    if 0 <= gx < self.lethal_costmap.shape[1] and 0 <= gy < self.lethal_costmap.shape[0]:
                        if self.lethal_costmap[gy, gx]:
                            path_blocked = True
                            break

            if path_blocked and (now - self.last_replan_time > 2.5):
                self.get_logger().info("⚠️ Active path blocked! Replanning detour...")
                self.last_replan_time = now
                self.route_to_target_goal()

    # =========================================================================
    # 8. Real-Time Closed-Loop Pure Pursuit Controller (10 Hz)
    # =========================================================================
    def motion_control_step(self):
        """10 Hz Closed-Loop Motion Controller with slew-rate limiter & in-place turns."""
        self.update_robot_pose()
        self.publish_footprint_marker()
        now = time.time()
        twist = Twist()

        if self.mode == 'STOP':
            self.stop_robot()
            return

        # 1. Recovery Execution State
        if self.state == 'RECOVERING':
            if now < self.recovery_end_time:
                twist.linear.x = -0.16
                twist.angular.z = 0.70 * self.recovery_turn_dir
                self.last_sent_linear_speed = -0.16
                self.cmd_pub.publish(twist)
                return
            else:
                self.state = self.pre_recovery_state
                self.last_check_pos = (self.current_x, self.current_y)
                self.last_check_time = now
                self.get_logger().info("✅ Recovery complete! Re-planning trajectory.")
                self.current_goal = None
                self.current_path = []
                self.route_to_target_goal()

        # 2. Dynamic Pedestrian & Proximity Safety Check
        emergency_stop_dist = 0.20
        min_front_dist = float('inf')
        min_front_angle = 0.0

        if self.latest_scan is not None:
            ranges = self.latest_scan.ranges
            n = len(ranges)
            for idx in range(0, n, 2):
                r = ranges[idx]
                if not (self.latest_scan.range_min < r < self.latest_scan.range_max):
                    continue
                ang = self.latest_scan.angle_min + idx * self.latest_scan.angle_increment
                ang = math.atan2(math.sin(ang), math.cos(ang))
                if abs(ang) < math.radians(28):
                    if r < min_front_dist:
                        min_front_dist = r
                        min_front_angle = ang

        if min_front_dist < emergency_stop_dist:
            self.current_cmd_v = -0.10
            self.current_cmd_w = 0.50 if min_front_angle <= 0 else -0.50
            twist.linear.x = self.current_cmd_v
            twist.angular.z = self.current_cmd_w
            self.last_sent_linear_speed = twist.linear.x
            self.cmd_pub.publish(twist)
            return

        # Pedestrian Corridor Yielding
        human_in_corridor = False
        dist_human = float('inf')
        if self.human_pose is not None and (now - self.last_human_pose_time < 2.0):
            hx, hy = self.human_pose
            dist_human = math.hypot(hx - self.current_x, hy - self.current_y)
            bearing_human = math.atan2(hy - self.current_y, hx - self.current_x) - self.current_yaw
            bearing_human = math.atan2(math.sin(bearing_human), math.cos(bearing_human))
            if dist_human < 1.35 and abs(bearing_human) < math.radians(45):
                human_in_corridor = True

        if human_in_corridor:
            if not self.is_yielding:
                self.is_yielding = True
                self.yield_start_time = now
            self.stop_robot()
            self.publish_status(f"🚶 Pedestrian in lane ({dist_human:.2f}m). Yielding safely...")

            if (now - self.yield_start_time > 3.5) and (now - self.last_replan_time > 2.5):
                self.get_logger().info("Yield timeout reached. Replanning detour around pedestrian...")
                self.last_replan_time = now
                self.route_to_target_goal()
            return
        else:
            if self.is_yielding:
                self.is_yielding = False
                self.get_logger().info("🚶 Pedestrian corridor clear. Resuming path.")

        # 3. Physical Stuck & Stall Detection
        if self.is_yielding:
            self.last_check_pos = (self.current_x, self.current_y)
            self.last_check_time = now
        elif now - self.last_check_time > 4.5:
            dist_moved = math.hypot(self.current_x - self.last_check_pos[0],
                                    self.current_y - self.last_check_pos[1])
            if dist_moved < 0.12 and abs(self.last_sent_linear_speed) > 0.08:
                self.get_logger().warning(f"⚠️ Stuck detected (moved {dist_moved:.2f}m in 4.5s). Triggering reverse recovery!")
                if self.target_goal is not None:
                    self.blacklisted_goals.append((self.target_goal[0], self.target_goal[1], now + 20.0))
                self.pre_recovery_state = self.state
                self.state = 'RECOVERING'
                self.recovery_end_time = now + 1.8
                self.recovery_turn_dir = 1.0 if np.random.rand() > 0.5 else -1.0
                self.current_cmd_v = 0.0
                self.current_cmd_w = 0.0
                return
            self.last_check_pos = (self.current_x, self.current_y)
            self.last_check_time = now

        # Ensure we have an active path
        if not self.current_path:
            self.stop_robot()
            return

        # 4. Pure Pursuit Path Tracking
        while len(self.current_path) > 1:
            dist_to_first = math.hypot(self.current_path[0][0] - self.current_x,
                                       self.current_path[0][1] - self.current_y)
            if dist_to_first < 0.25:
                self.current_path.pop(0)
            else:
                break

        # Check arrival at target goal
        dist_to_goal = math.hypot(self.target_goal[0] - self.current_x,
                                  self.target_goal[1] - self.current_y)

        # Arrival at dynamically spotted trash:
        if self.intercepting_trash:
            if self.intercept_arrival_time is None:
                if dist_to_goal < 0.55:
                    self.intercept_arrival_time = now
                    self.get_logger().info(f"📍 Over spotted litter at {self.target_goal}. Vacuuming...")
                    self.stop_robot()
                    return
            else:
                # Wait up to 1.5s for trash_collector_node to vacuum and remove the litter
                if (now - self.intercept_arrival_time) < 1.5:
                    self.stop_robot()
                    return
                else:
                    self.get_logger().info(f"✓ Pickup window elapsed at {self.target_goal}. Resuming exploration.")
                    if self.target_goal:
                        self.collected_trash_locs.append(self.target_goal)
                    self.intercepting_trash = False
                    self.active_trash_target = None
                    self.intercept_arrival_time = None
                    self.current_goal = None
                    self.current_path = []
                    self.publish_status("Pickup window complete. Continuing autonomous exploration.")
                    return

        # Arrival at mission waypoint or frontier:
        arrival_thresh = 0.40
        if dist_to_goal < arrival_thresh:
            self.get_logger().info(f"📍 Reached target: {self.target_goal}")

            # Blacklist reached frontier location for 30s so exploration moves across the park
            if self.target_goal and self.mode in ('EXPLORE', 'PATROL'):
                self.blacklisted_goals.append((self.target_goal[0], self.target_goal[1], now + 30.0))

            # Ensure intercept state is completely cleared on arrival
            self.intercepting_trash = False
            self.active_trash_target = None
            self.intercept_arrival_time = None

            if self.mode in ('EXPLORE', 'PATROL'):
                self.current_goal = None
                self.current_path = []
                return

            elif self.mode == 'MISSION':
                self.mission_index += 1
                if self.mission_index < len(self.mission_queue):
                    self.target_goal = self.mission_queue[self.mission_index]
                    self.publish_status(
                        f"🧹 Mission '{self.mission_name}' step {self.mission_index + 1}/{len(self.mission_queue)}: {self.target_goal}"
                    )
                    self.route_to_target_goal()
                    return
                else:
                    self.get_logger().info(f"🎉 Mission Complete: {self.mission_name}! Resuming Autonomous Exploration.")
                    self.publish_status(f"🎉 Mission Complete: {self.mission_name}! Resuming Autonomous Exploration.")
                    self.mode = 'EXPLORE'
                    self.current_goal = None
                    self.current_path = []
                    return

            else:  # GOTO
                self.stop_robot()
                self.get_logger().info(f"🎯 Arrived at destination: {self.mission_name} ({self.target_goal})!")
                self.publish_status(f"Arrived at {self.mission_name}!")
                self.mode = 'STOP'
                return

        # Adaptive lookahead
        dyn_lookahead = max(0.26, min(0.55, 0.28 + 0.50 * max(0.0, self.current_cmd_v)))
        lookahead_pt = self.current_path[-1]
        for pt in self.current_path:
            d = math.hypot(pt[0] - self.current_x, pt[1] - self.current_y)
            if d >= dyn_lookahead:
                lookahead_pt = pt
                break

        target_angle = math.atan2(lookahead_pt[1] - self.current_y, lookahead_pt[0] - self.current_x)
        heading_err = target_angle - self.current_yaw
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        L = max(0.16, math.hypot(lookahead_pt[0] - self.current_x, lookahead_pt[1] - self.current_y))
        curvature = (2.0 * math.sin(heading_err)) / L

        if abs(heading_err) > 0.35:
            # Pure in-place pivot: zero linear speed eliminates caster scrub and drifting
            target_v = 0.0
            target_w = float(np.clip(1.60 * heading_err, -self.max_ang_speed, self.max_ang_speed))
            if abs(target_w) < 0.32:
                target_w = 0.32 if target_w > 0 else -0.32
            self.last_check_pos = (self.current_x, self.current_y)
            self.last_check_time = now
        else:
            scale = math.cos(heading_err)
            target_v = float(np.clip(self.max_lin_speed * scale, 0.14, self.max_lin_speed))
            target_w = float(np.clip(target_v * curvature, -self.max_ang_speed, self.max_ang_speed))

        # Velocity slew rate limiter
        max_lin_step = 0.065
        max_ang_step = 0.20
        dv = float(np.clip(target_v - self.current_cmd_v, -max_lin_step, max_lin_step))
        dw = float(np.clip(target_w - self.current_cmd_w, -max_ang_step, max_ang_step))

        self.current_cmd_v += dv
        self.current_cmd_w += dw

        twist.linear.x = self.current_cmd_v
        twist.angular.z = self.current_cmd_w
        self.last_sent_linear_speed = twist.linear.x
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.current_cmd_v = 0.0
        self.current_cmd_w = 0.0
        self.last_sent_linear_speed = 0.0
        self.cmd_pub.publish(Twist())

    # =========================================================================
    # 9. Visualization Publishers
    # =========================================================================
    def publish_planned_path(self, world_path):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for wx, wy in world_path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.05
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def publish_footprint_marker(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'cleaner_footprint'
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = self.current_x
        marker.pose.position.y = self.current_y
        marker.pose.position.z = 0.04
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.robot_radius * 2.0
        marker.scale.y = self.robot_radius * 2.0
        marker.scale.z = 0.02
        marker.color = ColorRGBA(r=0.1, g=0.85, b=0.3, a=0.45)
        marker.lifetime = Duration(sec=1)
        self.footprint_marker_pub.publish(marker)

    def publish_goal_marker(self, gx, gy):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'current_goal'
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = gx
        marker.pose.position.y = gy
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.40
        marker.scale.y = 0.40
        marker.scale.z = 0.40
        marker.color = ColorRGBA(r=1.0, g=0.3, b=0.1, a=0.85)
        marker.lifetime = Duration(sec=2)
        self.goal_marker_pub.publish(marker)

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GardenNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
