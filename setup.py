import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'garden_cleaner_robot'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro') + glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jatin',
    maintainer_email='jatin@todo.todo',
    description='Autonomous Garden Cleaner Robot simulation package with Gazebo Harmonic and SLAM.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'trash_collector_node = garden_cleaner_robot.trash_collector_node:main',
            'garden_navigator = garden_cleaner_robot.garden_navigator:main',
            'pedestrian_simulator = garden_cleaner_robot.pedestrian_simulator:main',
            'trash_vision_detector = garden_cleaner_robot.trash_vision_detector:main',
            'slam_map_visualizer = garden_cleaner_robot.slam_map_visualizer:main',
            'slam_lifecycle_activator = garden_cleaner_robot.slam_lifecycle_activator:main',
            'cleaner_cli = garden_cleaner_robot.cleaner_cli:main',
        ],
    },
)
