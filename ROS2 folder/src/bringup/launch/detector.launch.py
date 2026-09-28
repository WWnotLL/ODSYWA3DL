import os

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from scipy.spatial.transform import Rotation


def lidar_rotation(detector_params):
    with open(detector_params, encoding='utf-8') as file:
        params = yaml.safe_load(file)['obstacle_detector']['ros__parameters']
    rotation = params['core']['preprocess']['axes']['rotation']
    return Rotation.from_matrix(np.reshape(rotation, (3, 3))).as_quat()


def detector_nodes(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    qx, qy, qz, qw = lidar_rotation(arg('detector_params'))
    nodes = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_mount',
            arguments=[
                '--frame-id', 'base_link', '--child-frame-id', arg('lidar_frame'),
                '--x', arg('lidar_x'), '--y', arg('lidar_y'), '--z', arg('lidar_z'),
                '--qx', str(qx), '--qy', str(qy), '--qz', str(qz), '--qw', str(qw),
            ],
        ),
        Node(
            package='obstacle_detector',
            executable='obstacle_detector_node',
            name='obstacle_detector',
            output='screen',
            parameters=[arg('detector_params'), {'pointcloud_topic': arg('pointcloud_topic')}],
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
        DeclareLaunchArgument('pointcloud_topic', default_value='/lidar_points',
                              description='Топик облака точек лидара'),
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
