import os
import xml.etree.ElementTree as ET
import pytest
from ament_index_python.packages import get_package_share_directory


def test_package_share_structure():
    share_dir = get_package_share_directory('garden_cleaner_robot')
    assert os.path.exists(os.path.join(share_dir, 'worlds', 'garden_park.sdf'))
    assert os.path.exists(os.path.join(share_dir, 'urdf', 'cleaner_robot.urdf.xacro'))
    assert os.path.exists(os.path.join(share_dir, 'launch', 'garden_simulation.launch.py'))
    assert os.path.exists(os.path.join(share_dir, 'config', 'slam_toolbox_params.yaml'))
    assert os.path.exists(os.path.join(share_dir, 'config', 'nav2_params.yaml'))
    assert os.path.exists(os.path.join(share_dir, 'config', 'garden_cleaner.rviz'))


def test_sdf_world_requirements():
    share_dir = get_package_share_directory('garden_cleaner_robot')
    sdf_path = os.path.join(share_dir, 'worlds', 'garden_park.sdf')

    tree = ET.parse(sdf_path)
    root = tree.getroot()
    world = root.find('world')
    assert world is not None
    assert world.attrib.get('name') == 'garden_park'

    # GPS Spherical Coordinates & NavSat Plugin Check
    spherical = world.find('spherical_coordinates')
    assert spherical is not None
    assert '37.774929' in spherical.find('latitude_deg').text
    assert '-122.419416' in spherical.find('longitude_deg').text

    world_plugins = [p.attrib.get('name') for p in world.findall('plugin')]
    assert 'gz::sim::systems::NavSat' in world_plugins

    models = {m.attrib.get('name'): m for m in world.findall('model')}

    # Ground plane check
    assert 'ground_plane' in models
    ground = models['ground_plane']
    ground_mat = ground.find('.//material')
    assert ground_mat is not None
    diffuse = ground_mat.find('diffuse').text.strip()
    ambient = ground_mat.find('ambient').text.strip()
    assert '0.15 0.55 0.15' in ambient
    assert '0.15 0.55 0.15' in diffuse

    # Pathways check (3 paths: a->b, b->c, c->a)
    assert 'garden_pathways' in models
    pathways = models['garden_pathways']
    links = [l.attrib.get('name') for l in pathways.findall('link')]
    assert 'path_a_to_b' in links
    assert 'path_b_to_c' in links
    assert 'path_c_to_a' in links

    # Hedges & Outer Boundary check
    assert 'garden_hedges' in models
    assert 'garden_outer_boundary' in models

    # 3 Static Obstacles check
    assert 'park_bench' in models
    assert 'tree_obstacle' in models
    assert 'planter_box' in models

    # Dynamic Obstacle check (garden_pedestrian with VelocityControl)
    assert 'garden_pedestrian' in models
    ped = models['garden_pedestrian']
    plugins = [p.attrib.get('name') for p in ped.findall('plugin')]
    assert 'gz::sim::systems::VelocityControl' in plugins

    # 6 Trash objects check (scattered on 3 paths)
    for t_name in ['trash_a1', 'trash_a2', 'trash_b1', 'trash_b2', 'trash_c1', 'trash_c2']:
        assert t_name in models, f"Model {t_name} missing from world"
