# Vishwakarma Awards: Autonomous Garden Cleaner Robot (`garden_cleaner_robot`)

A fully autonomous municipal cleaner and park explorer robot simulation built for the **Vishwakarma Awards** on **ROS 2 Lyrical** and **Gazebo Harmonic**.

---

## Key Features

1. **Cognitive Frontier Exploration (Zero Prior Knowledge)**:
   - Starts with **zero hardcoded waypoints or pre-known trash locations**.
   - Builds a real-time OccupancyGrid map via **SLAM Toolbox Online Async**.
   - Morphological boundary analysis detects unmapped territory frontiers.
   - Plans optimal, collision-free global trajectories to frontier centroids using in-memory dilated costmaps and A*.

2. **Onboard AI Computer Vision Trash Detector**:
   - Streams live RGB forward camera feed at 20 Hz (`/camera/image_raw`).
   - Uses multi-range OpenCV HSV color segmentation and geometric verification to detect:
     - **Crimson Red Beverage Cans**
     - **Sapphire Blue Drink Cans**
     - **Golden Yellow Snack Boxes**
   - Calculates 3D optical distance and camera bearing, projecting sightings into map coordinates (`/garden/detected_trash`).
   - Rejects sky/horizon artifacts and moving pedestrians using ground-plane ROI and area clamping.

3. **Dynamic Interception & Vacuum Collection Engine**:
   - When vision detects litter, navigator automatically diverts from frontier mapping to intercept the object.
   - Enters suction collection radius (`0.85m`), stops briefly over the litter, and vacuums it up.
   - Deletes the entity from Gazebo Harmonic in real-time, increments score (`+10 pts`), and updates the hopper capacity percentage (`/garden/bin_capacity_pct`).
   - Seamlessly resumes autonomous frontier exploration across the remaining park territory.

4. **100% Enclosed Park & Roundabout Navigation**:
   - Enclosed perimeter walls prevent any straying outside the garden boundaries.
   - Central roundabout features a raised natural stone wall and sculpted hedge crown ($360^\circ$ continuous LiDAR reflection at $z=0.315\text{m}$), preventing any robot stranding or collisions.

5. **Dual Desktop Pop-Up Windows**:
   - **AI Camera Vision Feed**: OpenCV desktop window displaying live camera imagery with corner-accented bounding boxes, 3D target coordinates, and distance tags.
   - **2D SLAM Map & Mission HUD**: Real-time dual-panel window showing the live 2D SLAM grid, trajectory trails, planned A* paths, robot heading $[R]$, dynamically discovered litter pins, GPS coordinates, speed, score, and hopper fill level.

6. **Dynamic Pedestrian Corridor Yielding**:
   - Humanoid visitor pacing smoothly along the north promenade.
   - Robot uses 360-degree LiDAR and corridor safety cones to proactively yield and maintain a safe clearance.

---

## Quick Launch & Commands

### 1. Launch Autonomous Simulation
Run the turnkey script:
```bash
/home/jatin/ros2_simulation_ws/src/garden_cleaner_robot/run_simulation.sh
```
*(Or via ROS 2 launch)*:
```bash
source /opt/ros/lyrical/setup.bash
source /home/jatin/ros2_simulation_ws/install/setup.bash
ros2 launch garden_cleaner_robot garden_simulation.launch.py autostart_patrol:=true
```

### 2. Interactive Mission Control CLI (Optional)
In a separate terminal, command the robot or inspect live telemetry:
```bash
source /opt/ros/lyrical/setup.bash
source /home/jatin/ros2_simulation_ws/install/setup.bash
ros2 run garden_cleaner_robot cleaner_cli
```
- Press `e`: **Autonomous Frontier Exploration**
- Press `c`: Clean Sector C to A
- Press `a`: Clean Sector A to B
- Press `b`: Clean Sector B to C
- Press `m`: Complete Triangle Sweep
- Press `1`, `2`, `3`: Navigate to Zone A, B, or C
- Press `s`: Pause / Stop Robot

### 3. Run Automated Package Tests
```bash
cd /home/jatin/ros2_simulation_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
pytest src/garden_cleaner_robot/test/ -v
```
*(Verifies URDF xacro generation, vacuum collection logic, mission state machines, visualizer nodes, share paths, and SDF requirements.)*
