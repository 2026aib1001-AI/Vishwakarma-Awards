import os
import subprocess
import pytest
from ament_index_python.packages import get_package_share_directory
from garden_cleaner_robot.trash_collector_node import TrashCollectorNode
from garden_cleaner_robot.garden_navigator import GardenNavigator
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import rclpy


@pytest.fixture(scope="module")
def ros_init():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_urdf_xacro_conversion():
    share_dir = get_package_share_directory('garden_cleaner_robot')
    xacro_path = os.path.join(share_dir, 'urdf', 'cleaner_robot.urdf.xacro')

    res = subprocess.run(['xacro', xacro_path], capture_output=True, text=True)
    assert res.returncode == 0
    urdf_content = res.stdout

    # Check drive wheels and caster
    assert 'left_wheel' in urdf_content
    assert 'right_wheel' in urdf_content
    assert 'caster_wheel' in urdf_content

    # Check planar lidar sensor & elevated mast
    assert '/scan' in urdf_content
    assert 'laser_frame' in urdf_content
    assert 'gpu_lidar' in urdf_content

    # Check RGB camera sensor
    assert 'camera_link' in urdf_content
    assert 'camera_optical_frame' in urdf_content
    assert '/camera/image_raw' in urdf_content

    # Check DiffDrive plugin
    assert 'gz::sim::systems::DiffDrive' in urdf_content
    assert '<topic>/cmd_vel</topic>' in urdf_content
    assert '<odom_topic>/odom</odom_topic>' in urdf_content

    # Check GPS / NavSat receiver module
    assert 'gps_link' in urdf_content
    assert '/gps/fix' in urdf_content
    assert 'navsat' in urdf_content


def test_trash_collector_logic(ros_init):
    node = TrashCollectorNode()

    # Initial state: 6 items scattered across the 3 paths
    assert node.score == 0
    assert len(node.trash_targets) == 6
    assert not node.trash_targets['trash_a1']['collected']

    # Simulate robot approaching trash_a1 (2.5, -3.2) within threshold
    odom = Odometry()
    odom.pose.pose.position.x = 2.48
    odom.pose.pose.position.y = -3.22
    node.odom_callback(odom)

    assert node.trash_targets['trash_a1']['collected'] is True
    assert node.score == 10

    # Simulate robot approaching trash_b1 (4.0, 6.0)
    odom.pose.pose.position.x = 4.02
    odom.pose.pose.position.y = 5.98
    node.odom_callback(odom)

    assert node.trash_targets['trash_b1']['collected'] is True
    assert node.score == 20

    # Simulate robot approaching trash_c1 (-7.2, 2.8)
    odom.pose.pose.position.x = -7.18
    odom.pose.pose.position.y = 2.82
    node.odom_callback(odom)

    assert node.trash_targets['trash_c1']['collected'] is True
    assert node.score == 30

    node.destroy_node()


def test_garden_navigator_missions(ros_init):
    node = GardenNavigator()

    # Test 'clean c to a' command
    msg = String()
    msg.data = 'clean c to a'
    node.goto_callback(msg)
    assert node.mode == 'MISSION'
    assert node.mission_name == "Clean Path C to A"
    assert node.target_goal == (-9.0, 6.0)

    # Test 'clean a to b' command
    msg.data = 'clean a to b'
    node.goto_callback(msg)
    assert node.mode == 'MISSION'
    assert node.mission_name == "Clean Path A to B"
    assert node.target_goal == (0.0, -7.0)

    # Test 'clean b to c' command
    msg.data = 'clean b to c'
    node.goto_callback(msg)
    assert node.mode == 'MISSION'
    assert node.mission_name == "Clean Path B to C"
    assert node.target_goal == (9.0, 6.0)

    # Test 'clean all' command
    msg.data = 'clean all'
    node.goto_callback(msg)
    assert node.mode == 'MISSION'
    assert "Clean-All" in node.mission_name or "Triangle" in node.mission_name

    # Test 'go to a' command
    msg.data = 'go to a'
    node.goto_callback(msg)
    assert node.mode == 'GOTO'
    assert node.target_goal == (0.0, -7.0)

    # Test 'go to b' command
    msg.data = 'go to b'
    node.goto_callback(msg)
    assert node.mode == 'GOTO'
    assert node.target_goal == (9.0, 6.0)

    # Test 'go to c' command
    msg.data = 'go to c'
    node.goto_callback(msg)
    assert node.mode == 'GOTO'
    assert node.target_goal == (-9.0, 6.0)

    # Test 'patrol' command
    msg.data = 'patrol'
    node.goto_callback(msg)
    assert node.mode == 'PATROL'

    # Test 'stop' command
    msg.data = 'stop'
    node.goto_callback(msg)
    assert node.mode == 'STOP'

    # Test A* planner fallback and interpolation
    start_pt = (0.0, -7.0)
    goal_pt = (9.0, 6.0)
    path = node.plan_astar_path_with_fallback(start_pt, goal_pt)
    assert path is not None
    assert len(path) > 10  # Dense waypoints spaced <= 0.15m
    assert abs(path[0][0] - start_pt[0]) < 0.2
    assert abs(path[-1][0] - goal_pt[0]) < 0.2

    node.destroy_node()


def test_slam_map_visualizer_node(ros_init):
    from garden_cleaner_robot.slam_map_visualizer import SlamMapVisualizer
    node = SlamMapVisualizer()
    assert node is not None
    assert node.map_w == 720
    assert len(node.trash_targets) == 6

    # Test render loop executes without crash on dummy canvas
    node.render_loop()
    node.destroy_node()

