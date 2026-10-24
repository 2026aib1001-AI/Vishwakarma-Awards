#!/usr/bin/env bash
# ==============================================================================
# Garden Cleaner Robot: Complete Polished Simulation Bringup Script
# ==============================================================================
set -e

source /opt/ros/lyrical/setup.bash
WS_DIR="/home/jatin/ros2_simulation_ws"
if [ -f "${WS_DIR}/install/setup.bash" ]; then
    source "${WS_DIR}/install/setup.bash"
fi

# Clean up any lingering simulation processes
killall -9 gz-sim-main async_slam_toolbox_node python3 2>/dev/null || true
sleep 1

echo "======================================================================"
echo "  🌿 STARTING AUTONOMOUS GARDEN CLEANER ROBOT SIMULATION (ROS 2)      "
echo "======================================================================"
echo "  • World: Enclosed Garden Park with Sculpted Roundabout"
echo "  • Exploration: Cognitive Frontier Discovery (Zero Prior Knowledge)"
echo "  • Vision: Real-Time RGB Camera Object Detector (3D Spatial Mapping)"
echo "  • Cleaning: Autonomous Dynamic Suction Vacuum with Live Telemetry"
echo "  • Windows: Desktop Pop-Ups (Live AI Vision HUD & 2D SLAM Map HUD)"
echo "======================================================================"

ros2 launch garden_cleaner_robot garden_simulation.launch.py autostart_patrol:=true "$@"
