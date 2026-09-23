import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('bringup')
    default_rviz = os.path.join(pkg_share, 'config', 'lidar_monitor.rviz')
    default_bags_dir = '/workspace/data/for_hackathon'

    bag_arg = DeclareLaunchArgument(
        'bag', default_value='doubleT_obstacle',
        description='Имя папки бэга внутри data/for_hackathon/',
    )
    bags_dir_arg = DeclareLaunchArgument(
        'bags_dir', default_value=default_bags_dir,
        description='Путь к папке с распакованными бэгами',
    )
    rate_arg = DeclareLaunchArgument('rate', default_value='1.0', description='Скорость проигрывания')
    loop_arg = DeclareLaunchArgument('loop', default_value='true', description='Зациклить проигрывание')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true', description='Открывать ли RViz')
    rviz_config_arg = DeclareLaunchArgument('rviz_config', default_value=default_rviz)

    bag_path = PathJoinSubstitution([LaunchConfiguration('bags_dir'), LaunchConfiguration('bag')])

    bag_play_loop = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--rate', LaunchConfiguration('rate'), '--loop'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('loop')),
    )
    bag_play_once = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--rate', LaunchConfiguration('rate')],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('loop')),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    tf_lidar_livox = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_lidar_livox',
        arguments=['--frame-id', 'map', '--child-frame-id', 'lidar_livox'],
        output='screen',
    )
    tf_hesai_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_hesai_lidar',
        arguments=['--frame-id', 'map', '--child-frame-id', 'hesai_lidar'],
        output='screen',
    )

    return LaunchDescription([
        bag_arg,
        bags_dir_arg,
        rate_arg,
        loop_arg,
        rviz_arg,
        rviz_config_arg,
        tf_lidar_livox,
        tf_hesai_lidar,
        bag_play_loop,
        bag_play_once,
        rviz_node,
    ])
