import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('garden_cleaner_robot')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    slam_dir = get_package_share_directory('slam_toolbox')
    nav2_dir = get_package_share_directory('nav2_bringup')

    world_file = os.path.join(pkg_share, 'worlds', 'garden_park.sdf')
    xacro_file = os.path.join(pkg_share, 'urdf', 'cleaner_robot.urdf.xacro')
    slam_params = os.path.join(pkg_share, 'config', 'slam_toolbox_params.yaml')
    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    rviz_config = os.path.join(pkg_share, 'config', 'garden_cleaner.rviz')

    # Ensure Gazebo Harmonic can locate mesh and world resources
    pkg_share_parent = os.path.dirname(pkg_share)
    if 'GZ_SIM_RESOURCE_PATH' in os.environ:
        os.environ['GZ_SIM_RESOURCE_PATH'] += ':' + pkg_share_parent + ':' + pkg_share
    else:
        os.environ['GZ_SIM_RESOURCE_PATH'] = pkg_share_parent + ':' + pkg_share

    # Launch Configurations & Arguments
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to launch RViz2 (SLAM Map pop-up)'
    )

    declare_vision_cmd = DeclareLaunchArgument(
        'vision_popup',
        default_value='true',
        description='Whether to show live OpenCV AI Camera Vision feed pop-up'
    )

    declare_map_popup_cmd = DeclareLaunchArgument(
        'map_popup',
        default_value='true',
        description='Whether to show live OpenCV SLAM Map pop-up window'
    )

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart_patrol',
        default_value='true',
        description='Whether to autonomously start garden patrol upon launch'
    )

    declare_use_nav2_cmd = DeclareLaunchArgument(
        'use_nav2',
        default_value='false',
        description='Whether to launch Nav2 (Default: false, using proven Cognitive A* + Pure Pursuit)'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('rviz')
    show_vision_popup = LaunchConfiguration('vision_popup')
    show_map_popup = LaunchConfiguration('map_popup')
    autostart_patrol = LaunchConfiguration('autostart_patrol')
    use_nav2 = LaunchConfiguration('use_nav2')

    # 1. Gazebo Harmonic with Expanded 3-Path Garden Park World
    gz_sim_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -v1 ', world_file]}.items()
    )

    # 2. Robot State Publisher (URDF + TF tree + Camera link)
    robot_desc = Command(['xacro ', xacro_file])
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time
        }]
    )

    # 3. Spawn Cleaner Robot at Zone A (x=0.0, y=-7.0, z=0.08, facing Path A->B)
    spawn_robot = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                output='screen',
                arguments=[
                    '-world', 'garden_park',
                    '-name', 'cleaner_robot',
                    '-topic', 'robot_description',
                    '-x', '0.0',
                    '-y', '-7.0',
                    '-z', '0.08',
                    '-Y', '0.957'
                ]
            )
        ]
    )

    # 4. ROS-Gazebo Parameter Bridge (Includes Camera, LiDAR, Odometry, TF, Twist, Delete Service)
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/pedestrian/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/gps/fix@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat',
            '/world/garden_park/remove@ros_gz_interfaces/srv/DeleteEntity@gz.msgs.Entity@gz.msgs.Boolean',
            '/world/garden_park/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 5. SLAM Toolbox Online Async Mapping (Configured & Activated)
    slam_cmd = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_dir, 'launch', 'online_async_launch.py')
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'slam_params_file': slam_params,
                }.items()
            )
        ]
    )

    # 5b. SLAM Lifecycle Auto-Activator (Guarantees SLAM transitions to 'active' [3] and publishes /map)
    slam_activator_node = TimerAction(
        period=4.2,
        actions=[
            Node(
                package='garden_cleaner_robot',
                executable='slam_lifecycle_activator',
                name='slam_lifecycle_activator',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )

    # 5c. Nav2 Navigation Stack (Controller, Planner, Behaviors, BT Navigator, Lifecycle Manager)
    nav2_nodes = TimerAction(
        period=5.2,
        actions=[
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                remappings=[('cmd_vel', '/cmd_vel_nav')],
                condition=IfCondition(use_nav2)
            ),
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                condition=IfCondition(use_nav2)
            ),
            Node(
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                remappings=[('cmd_vel', '/cmd_vel_nav')],
                condition=IfCondition(use_nav2)
            ),
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                condition=IfCondition(use_nav2)
            ),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'autostart': True,
                    'node_names': [
                        'controller_server',
                        'planner_server',
                        'behavior_server',
                        'bt_navigator',
                    ]
                }],
                condition=IfCondition(use_nav2)
            ),
        ]
    )

    # 6. Trash Collector & Scoring Node
    trash_collector_node = TimerAction(
        period=3.8,
        actions=[
            Node(
                package='garden_cleaner_robot',
                executable='trash_collector_node',
                name='trash_collector_node',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'proximity_threshold': 0.85,
                    'world_name': 'garden_park'
                }]
            )
        ]
    )

    # 7. AI Trash Vision Detector (Camera Feed Pop-up with OpenCV GUI)
    vision_detector_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='garden_cleaner_robot',
                executable='trash_vision_detector',
                name='trash_vision_detector',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'show_window': show_vision_popup
                }]
            )
        ]
    )

    # 7b. Real-Time SLAM Map Pop-Up Visualizer Window (with Mission HUD Sidebar)
    map_visualizer_node = TimerAction(
        period=4.5,
        actions=[
            Node(
                package='garden_cleaner_robot',
                executable='slam_map_visualizer',
                name='slam_map_visualizer',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'show_window': show_map_popup
                }]
            )
        ]
    )

    # 8. Dynamic Moving Pedestrian Controller
    pedestrian_node = TimerAction(
        period=3.8,
        actions=[
            Node(
                package='garden_cleaner_robot',
                executable='pedestrian_simulator',
                name='pedestrian_simulator',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'walking_speed': 0.45
                }]
            )
        ]
    )

    # 9. 3-Path Autonomous Road Navigator (Supports "clean c to a", "clean a to b", etc.)
    navigator_node = TimerAction(
        period=5.5,
        actions=[
            Node(
                package='garden_cleaner_robot',
                executable='garden_navigator',
                name='garden_navigator',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'autostart_patrol': autostart_patrol,
                    'use_nav2': use_nav2,
                    'max_linear_speed': 0.40,
                    'max_angular_speed': 0.85
                }]
            )
        ]
    )

    # 10. RViz2 Visualization (SLAM Map & Nav2 Pop-up)
    rviz_node = TimerAction(
        period=4.8,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
                parameters=[{'use_sim_time': use_sim_time}],
                condition=IfCondition(use_rviz)
            )
        ]
    )

    return LaunchDescription([
        declare_rviz_cmd,
        declare_vision_cmd,
        declare_map_popup_cmd,
        declare_autostart_cmd,
        declare_use_nav2_cmd,
        declare_use_sim_time_cmd,
        gz_sim_cmd,
        rsp_node,
        spawn_robot,
        bridge_node,
        slam_cmd,
        slam_activator_node,
        nav2_nodes,
        trash_collector_node,
        vision_detector_node,
        map_visualizer_node,
        pedestrian_node,
        navigator_node,
        rviz_node,
    ])
