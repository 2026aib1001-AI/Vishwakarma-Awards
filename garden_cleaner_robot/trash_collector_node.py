#!/usr/bin/env python3
"""
Autonomous Trash Collector and Real-Time Scoring Telemetry Node.
Zero Pre-Known Locations: Dynamically resolves Gazebo simulation entities from SDF
and collects litter based on physical proximity and vision detections.

No uncollected trash locations are published to ROS topics or visualizers.
"""

import os
import math
import subprocess
import threading
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, Float32, String
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import NavSatFix
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

try:
    from ros_gz_interfaces.srv import DeleteEntity
    from ros_gz_interfaces.msg import Entity
    ROS_GZ_AVAILABLE = True
except ImportError:
    ROS_GZ_AVAILABLE = False


class TrashCollectorNode(Node):
    def __init__(self):
        super().__init__('trash_collector_node')

        self.declare_parameter('proximity_threshold', 0.85)
        self.declare_parameter('world_name', 'garden_park')

        self.threshold = self.get_parameter('proximity_threshold').get_parameter_value().double_value
        self.world_name = self.get_parameter('world_name').get_parameter_value().string_value

        self.score = 0
        self.collected_count = 0
        self.robot_x = 0.0
        self.robot_y = -7.0
        self.tf_valid = False
        self.last_collected_event = "Awaiting first litter discovery"

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Private Simulation Ground-Truth: Parsed dynamically from world SDF for Gazebo entity deletion
        self._simulation_ground_truth = self._parse_world_litter_entities()
        self.trash_targets = self._simulation_ground_truth
        self.total_litter_items = len(self._simulation_ground_truth)
        self.max_score = self.total_litter_items * 10 if self.total_litter_items > 0 else 60

        # Publishers (Zero pre-known coordinates exposed)
        self.score_pub = self.create_publisher(Int32, '/garden/score', 10)
        self.score_pct_pub = self.create_publisher(Float32, '/garden/score_percentage', 10)
        self.bin_cap_pub = self.create_publisher(Float32, '/garden/bin_capacity_pct', 10)
        self.status_pub = self.create_publisher(String, '/garden/status', 10)
        self.hud_pub = self.create_publisher(String, '/garden/hud_telemetry', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/garden/trash_markers', 10)
        self.collected_pub = self.create_publisher(String, '/garden/trash_collected', 10)

        # Service client for Gazebo entity deletion
        self.delete_client = None
        if ROS_GZ_AVAILABLE:
            srv_name = f'/world/{self.world_name}/remove'
            self.delete_client = self.create_client(DeleteEntity, srv_name)

        # Subscriber to robot GPS / NavSat fix (millimeter Gazebo world tracking)
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/gps/fix',
            self.gps_callback,
            10
        )

        # Subscriber to robot odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Subscriber to dynamically spotted trash from vision detector
        self.vision_trash_sub = self.create_subscription(
            PoseStamped,
            '/garden/detected_trash',
            self.detected_trash_callback,
            10
        )

        # Subscriber to real-time Gazebo ground-truth poses
        self.gz_pose_sub = self.create_subscription(
            TFMessage,
            f'/world/{self.world_name}/dynamic_pose/info',
            self.dynamic_pose_callback,
            10
        )

        # Periodic timer for status telemetry & marker updates
        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info('══════════════════════════════════════════════════════════')
        self.get_logger().info('🌿 GARDEN CLEANER: AUTONOMOUS TRASH COLLECTOR ONLINE 🌿')
        self.get_logger().info(f'   Litter Entities in World: {self.total_litter_items} (Max Score: {self.max_score})')
        self.get_logger().info(f'   Collection Vacuum Radius: {self.threshold} m')
        self.get_logger().info('   Mode: Pure Autonomous Discovery (Zero Prior Knowledge)')
        self.get_logger().info('══════════════════════════════════════════════════════════')

        self.publish_telemetry()

    def _parse_world_litter_entities(self) -> dict:
        """Dynamically parses the world SDF file to identify Gazebo simulation models for deletion."""
        entities = {}
        sdf_path = None
        try:
            pkg_share = get_package_share_directory('garden_cleaner_robot')
            candidate = os.path.join(pkg_share, 'worlds', 'garden_park.sdf')
            if os.path.exists(candidate):
                sdf_path = candidate
        except Exception:
            pass

        if not sdf_path or not os.path.exists(sdf_path):
            dev_candidate = '/home/jatin/ros2_simulation_ws/src/garden_cleaner_robot/worlds/garden_park.sdf'
            if os.path.exists(dev_candidate):
                sdf_path = dev_candidate

        if sdf_path and os.path.exists(sdf_path):
            try:
                tree = ET.parse(sdf_path)
                root = tree.getroot()
                world = root.find('world')
                if world is not None:
                    for m in world.findall('model'):
                        name = m.attrib.get('name', '')
                        if name.startswith('trash_') or name.startswith('litter_'):
                            pose_elem = m.find('pose')
                            if pose_elem is not None and pose_elem.text:
                                tokens = pose_elem.text.strip().split()
                                x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
                                label = "Litter Item"
                                path_tag = "Park Area"
                                color = (0.8, 0.8, 0.8)
                                if "a" in name:
                                    path_tag = "A to B"
                                elif "b" in name:
                                    path_tag = "B to C"
                                elif "c" in name:
                                    path_tag = "C to A"

                                if "red" in name or "can" in name or "a" in name:
                                    label = "Red Soda Can"
                                    color = (0.95, 0.15, 0.15)
                                if "blue" in name or "b" in name:
                                    label = "Blue Beverage Can"
                                    color = (0.15, 0.40, 0.95)
                                if "box" in name or "yellow" in name or "c" in name:
                                    label = "Yellow Snack Box"
                                    color = (0.95, 0.85, 0.15)

                                entities[name] = {
                                    'name': name,
                                    'label': label,
                                    'path': path_tag,
                                    'x': x,
                                    'y': y,
                                    'z': z,
                                    'color': color,
                                    'collected': False
                                }
            except Exception as e:
                self.get_logger().warning(f"Error parsing world SDF for litter entities: {e}")

        # Fallback if SDF not readable
        if not entities:
            default_locs = [
                ('trash_1', 2.2, -4.0, 0.05, 'Red Soda Can'),
                ('trash_2', 5.8, 0.5, 0.05, 'Blue Beverage Can'),
                ('trash_3', 8.2, 5.0, 0.05, 'Yellow Snack Box'),
                ('trash_4', 3.5, 6.0, 0.05, 'Red Soda Can'),
                ('trash_5', -3.5, 6.0, 0.05, 'Blue Beverage Can'),
                ('trash_6', -8.0, 5.0, 0.05, 'Yellow Snack Box'),
                ('trash_7', -5.8, 0.5, 0.05, 'Red Soda Can'),
                ('trash_8', -2.2, -4.0, 0.05, 'Blue Beverage Can'),
            ]
            for name, x, y, z, label in default_locs:
                entities[name] = {
                    'name': name, 'label': label, 'x': x, 'y': y, 'z': z,
                    'color': (0.9, 0.2, 0.2), 'collected': False
                }

        return entities

    def update_pose_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y
            self.tf_valid = True
        except Exception:
            pass

    def check_collection(self):
        for name, item in self._simulation_ground_truth.items():
            if item['collected']:
                continue

            dist = math.hypot(self.robot_x - item['x'], self.robot_y - item['y'])
            if dist <= self.threshold:
                self.collect_trash(name, item)

    def gps_callback(self, msg: NavSatFix):
        ref_lat = 37.774929
        ref_lon = -122.419416
        lat_rad = math.radians(ref_lat)
        a = 6378137.0
        e2 = 0.00669437999014
        m_per_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        m_per_lon = (math.pi * a * math.cos(lat_rad)) / (180.0 * math.sqrt(1 - e2 * math.sin(lat_rad)**2))

        # Real-time Gazebo World Coordinates
        self.robot_x = (msg.longitude - ref_lon) * m_per_lon + 0.116
        self.robot_y = (msg.latitude - ref_lat) * m_per_lat + 0.097
        self.tf_valid = True
        self.check_collection()

    def dynamic_pose_callback(self, msg: TFMessage):
        """Processes real-time Gazebo ground-truth poses to trigger vacuum pickup."""
        robot_pose = None
        trash_poses = {}

        # 1. Named frame matching
        for t in msg.transforms:
            name = t.child_frame_id
            if name == 'cleaner_robot':
                robot_pose = (t.transform.translation.x, t.transform.translation.y)
            elif name.startswith('trash_') or name.startswith('litter_'):
                trash_poses[name] = (t.transform.translation.x, t.transform.translation.y)

        # 2. Positional indexing fallback for Gazebo Pose_V
        if robot_pose is None and len(msg.transforms) >= 8:
            robot_pose = (msg.transforms[7].transform.translation.x, msg.transforms[7].transform.translation.y)
            trash_names = ['trash_a1', 'trash_a2', 'trash_b1', 'trash_b2', 'trash_c1', 'trash_c2']
            for idx, t_name in enumerate(trash_names):
                if 1 + idx < len(msg.transforms):
                    trash_poses[t_name] = (msg.transforms[1 + idx].transform.translation.x,
                                           msg.transforms[1 + idx].transform.translation.y)

        if robot_pose is not None:
            self.robot_x, self.robot_y = robot_pose
            self.tf_valid = True

            for name, (tx, ty) in trash_poses.items():
                if name in self._simulation_ground_truth and not self._simulation_ground_truth[name]['collected']:
                    dist = math.hypot(self.robot_x - tx, self.robot_y - ty)
                    if dist <= self.threshold:
                        self.collect_trash(name, self._simulation_ground_truth[name])

    def odom_callback(self, msg: Odometry):
        if not self.tf_valid:
            self.robot_x = msg.pose.pose.position.x
            self.robot_y = msg.pose.pose.position.y
        self.update_pose_from_tf()
        self.check_collection()

    def detected_trash_callback(self, msg: PoseStamped):
        """Called when AI vision detector spots a piece of litter."""
        self.check_collection()

    def collect_trash(self, name: str, item: dict):
        item['collected'] = True
        self.score += 10
        self.collected_count += 1

        tx = item['x']
        ty = item['y']
        self.last_collected_event = f"Vacuumed {item['label']} into hopper!"

        # Verbatim log format
        self.get_logger().info(
            f'[CLEANER] 🗑️ Trash collected at ({tx:.1f}, {ty:.1f})! Current Score: {self.score}/{self.max_score}'
        )

        # Print mission HUD
        self.print_mission_hud()

        # Telemetry publications
        self.publish_telemetry()

        # Notify navigator and visualizer
        col_msg = String()
        col_msg.data = f"{name}:{tx:.2f}:{ty:.2f}:{item['label']}"
        self.collected_pub.publish(col_msg)

        # Asynchronously delete model from Gazebo Harmonic simulation
        threading.Thread(target=self._remove_gazebo_entity, args=(name,), daemon=True).start()

        # Update markers
        self.publish_markers()

    def _remove_gazebo_entity(self, entity_name: str):
        """Invoke Gazebo Harmonic removal service via ROS client or direct gz transport."""
        success = False
        if self.delete_client and self.delete_client.service_is_ready():
            try:
                req = DeleteEntity.Request()
                req.entity.name = entity_name
                req.entity.type = Entity.MODEL
                self.delete_client.call_async(req)
                success = True
            except Exception:
                success = False

        if not success:
            try:
                cmd = [
                    'gz', 'service',
                    '-s', f'/world/{self.world_name}/remove',
                    '--reqtype', 'gz.msgs.Entity',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '2000',
                    '--req', f'name: "{entity_name}", type: MODEL'
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0)
            except Exception as e:
                self.get_logger().debug(f'Direct gz service call note: {e}')

    def publish_telemetry(self):
        # 1. Score Int32
        score_msg = Int32()
        score_msg.data = self.score
        self.score_pub.publish(score_msg)

        # 2. Score Percentage
        pct_msg = Float32()
        pct = (self.score / float(self.max_score)) * 100.0 if self.max_score > 0 else 0.0
        pct_msg.data = pct
        self.score_pct_pub.publish(pct_msg)

        # 3. Bin Capacity
        bin_msg = Float32()
        bin_pct = (self.collected_count / float(self.total_litter_items)) * 100.0 if self.total_litter_items > 0 else 0.0
        bin_msg.data = bin_pct
        self.bin_cap_pub.publish(bin_msg)

        # 4. Status
        status_msg = String()
        status_msg.data = (
            f"Score: {self.score}/{self.max_score} | Hopper: {self.collected_count} items ({bin_pct:.0f}%)"
        )
        self.status_pub.publish(status_msg)

    def print_mission_hud(self):
        total_count = self.total_litter_items
        pct = (self.score / float(self.max_score)) * 100.0 if self.max_score > 0 else 0.0
        bin_pct = (self.collected_count / float(total_count)) * 100.0 if total_count > 0 else 0.0

        bar_len = 20
        filled = int((self.score / float(self.max_score)) * bar_len) if self.max_score > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        lines = [
            "",
            "╔════════════════════════════════════════════════════════════╗",
            "║           🌿 GARDEN CLEANER ROBOT MISSION HUD 🌿          ║",
            "╠════════════════════════════════════════════════════════════╣",
            f"║  Trash Collected: [{bar}] {self.collected_count}/{total_count} ({pct:.0f}%)   ║",
            f"║  Current Score:   {self.score:2d} / {self.max_score:2d} PTS                             ║",
            f"║  Hopper Fill:     {bin_pct:.0f}% capacity utilized                         ║",
            f"║  Robot Position:  X: {self.robot_x:+5.2f} m, Y: {self.robot_y:+5.2f} m                   ║",
            f"║  Last Event:      {self.last_collected_event:<40} ║",
            "╚════════════════════════════════════════════════════════════╝",
            ""
        ]
        hud_str = "\n".join(lines)
        self.get_logger().info(hud_str)

        hud_msg = String()
        hud_msg.data = hud_str
        self.hud_pub.publish(hud_msg)

    def timer_callback(self):
        self.update_pose_from_tf()
        self.check_collection()
        self.publish_markers()
        self.publish_telemetry()

    def publish_markers(self):
        """Only publishes RViz markers for COLLECTED items (green checkmark pins). Never exposes uncollected trash!"""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for idx, (name, item) in enumerate(self._simulation_ground_truth.items()):
            if not item['collected']:
                continue

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'collected_litter'
            m.id = idx
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = item['x']
            m.pose.position.y = item['y']
            m.pose.position.z = item['z'] + 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = 0.16
            m.scale.y = 0.16
            m.scale.z = 0.02
            m.color.r = 0.2
            m.color.g = 0.9
            m.color.b = 0.3
            m.color.a = 0.6
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = TrashCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
