import os

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from scipy.spatial.transform import Rotation

LIDAR_MOUNTS = {
    'hesai_lidar': 1.081,
    'lidar_livox': 1.71,
}


def lidar_rotation(detector_params):
    with open(detector_params, encoding='utf-8') as file:
        params = yaml.safe_load(file)['obstacle_detector']['ros__parameters']
    rotation = params['core']['preprocess']['axes']['rotation']
    return Rotation.from_matrix(np.reshape(rotation, (3, 3))).as_quat()


def detector_nodes(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    qx, qy, qz, qw = lidar_rotation(arg('detector_params'))
    parameters = [arg('detector_params')]
    if arg('pointcloud_topic'):
        parameters.append({'pointcloud_topics': [arg('pointcloud_topic')]})
    mounts = {frame: ('0.0', '0.0', str(z)) for frame, z in LIDAR_MOUNTS.items()}
    mounts[arg('lidar_frame')] = (arg('lidar_x'), arg('lidar_y'), arg('lidar_z'))
    nodes = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'lidar_mount_{frame}',
            arguments=[
                '--frame-id', 'base_link', '--child-frame-id', frame,
                '--x', x, '--y', y, '--z', z,
                '--qx', str(qx), '--qy', str(qy), '--qz', str(qz), '--qw', str(qw),
            ],
        )
        for frame, (x, y, z) in mounts.items()
    ]
    nodes += [
        Node(
            package='obstacle_detector',
            executable='obstacle_detector_node',
            name='obstacle_detector',
            output='screen',
            parameters=parameters,
            additional_env={'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'},
        ),
    ]
    if arg('bag'):
        nodes.append(ExecuteProcess(
            cmd=['ros2', 'bag', 'play', arg('bag'), '--delay', '3'],
            output='screen',
        ))
    return nodes


def generate_launch_description():
    detector_params = os.path.join(
        get_package_share_directory('obstacle_detector'), 'config', 'obstacle_detector.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('pointcloud_topic', default_value='',
                              description='Топик облака лидара; пусто — топики из конфига'),
        DeclareLaunchArgument('lidar_frame', default_value='hesai_lidar',
                              description='frame_id облака лидара'),
        DeclareLaunchArgument('lidar_x', default_value='0.0', description='Лидар в base_link: вперёд, м'),
        DeclareLaunchArgument('lidar_y', default_value='0.0', description='Лидар в base_link: влево, м'),
        DeclareLaunchArgument('lidar_z', default_value='1.081',
                              description='Лидар в base_link: над головкой рельса, м'),
        DeclareLaunchArgument('bag', default_value='',
                              description='Путь к записи rosbag2 для проигрывания; пусто — ждать данные от лидара'),
        DeclareLaunchArgument('detector_params', default_value=detector_params,
                              description='Конфиг ноды и ядра; из него же берётся поворот лидара'),
        OpaqueFunction(function=detector_nodes),
    ])
