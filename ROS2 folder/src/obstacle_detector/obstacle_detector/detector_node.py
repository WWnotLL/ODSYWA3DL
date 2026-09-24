import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Quaternion, Vector3
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, ColorRGBA, Float32
from visualization_msgs.msg import Marker, MarkerArray

from core.config import Config
from core.pipeline import ObstacleDetector

NUMPY_TYPES = {
    PointField.INT8: '<i1', PointField.UINT8: '<u1',
    PointField.INT16: '<i2', PointField.UINT16: '<u2',
    PointField.INT32: '<i4', PointField.UINT32: '<u4',
    PointField.FLOAT32: '<f4', PointField.FLOAT64: '<f8',
}
NULLABLE_CORE_PARAMETERS = ('gauge.y_center_offset_m', 'gauge.axis_slope', 'detector.max_range_m')
MIN_BOX_SIZE_M = 0.05
BOX_COLOR = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.5)


def pointcloud_to_arrays(msg):
    point_type = np.dtype({
        'names': [field.name for field in msg.fields],
        'formats': [NUMPY_TYPES[field.datatype] for field in msg.fields],
        'offsets': [field.offset for field in msg.fields],
        'itemsize': msg.point_step,
    })
    points = np.frombuffer(msg.data, dtype=point_type, count=msg.width * msg.height)
    xyz = np.stack([points['x'], points['y'], points['z']], axis=1)
    return xyz, points['intensity'], points['ring'], points['timestamp']


def stamp_to_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def flatten(section, prefix=''):
    for key, value in section.items():
        if isinstance(value, dict):
            yield from flatten(value, f'{prefix}{key}.')
        else:
            yield f'{prefix}{key}', value


def set_nested(settings, dotted_name, value):
    *sections, key = dotted_name.split('.')
    for section in sections:
        settings = settings.setdefault(section, {})
    settings[key] = value


class ObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        defaults_path = get_package_share_directory('obstacle_detector') + '/config/params.yaml'
        with open(defaults_path, encoding='utf-8') as file:
            defaults = yaml.safe_load(file)['obstacle_detector']['ros__parameters']

        pointcloud_topic = self.declare_parameter('pointcloud_topic', defaults['pointcloud_topic']).value
        self.lidar_period_ns = int(self.declare_parameter('lidar_period_s', defaults['lidar_period_s']).value * 1e9)
        self.record_jump_ns = int(self.declare_parameter('record_jump_s', defaults['record_jump_s']).value * 1e9)

        self.detector = ObstacleDetector(self.declare_core_config(defaults['core']))
        self.previous_stamp_ns = None

        self.checked_pub = self.create_publisher(Bool, '/obstacle/checked', 10)
        self.detected_pub = self.create_publisher(Bool, '/obstacle/detected', 10)
        self.distance_pub = self.create_publisher(Float32, '/obstacle/distance', 10)
        self.stop_pub = self.create_publisher(Bool, '/obstacle/stop', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/obstacle/markers', 10)
        self.create_subscription(PointCloud2, pointcloud_topic, self.handle_pointcloud, 10)

        self.get_logger().info(f'Listening on {pointcloud_topic}')

    def declare_core_config(self, core_defaults):
        settings = {}
        for name, default in flatten(core_defaults):
            set_nested(settings, name, self.declare_parameter(f'core.{name}', default).value)
        for name in NULLABLE_CORE_PARAMETERS:
            set_nested(settings, name, self.declare_parameter(f'core.{name}', Parameter.Type.DOUBLE).value)

        axes = settings['preprocess']['axes']
        axes['rotation'] = np.reshape(axes['rotation'], (3, 3)).tolist()
        settings['gauge']['profile_mm'] = np.reshape(settings['gauge']['profile_mm'], (-1, 2)).tolist()
        return Config.from_dict(settings)

    def handle_pointcloud(self, msg):
        stamp_ns = stamp_to_ns(msg.header.stamp)
        frame_gap = self.frame_gap_since_previous(stamp_ns)
        xyz, intensity, ring, point_times = pointcloud_to_arrays(msg)
        result = self.detector.process(xyz, intensity, ring, point_times, stamp_ns, frame_gap=frame_gap)

        checked = result.debug['checked']
        stop = result.obstacle_found or (checked and result.debug['verdict']['stop'])
        distance = result.nearest_distance_m

        self.checked_pub.publish(Bool(data=checked))
        self.detected_pub.publish(Bool(data=result.obstacle_found))
        self.distance_pub.publish(Float32(data=float('nan') if distance is None else distance))
        self.stop_pub.publish(Bool(data=stop))
        self.markers_pub.publish(self.obstacle_markers(msg.header, result))

    def frame_gap_since_previous(self, stamp_ns):
        previous_stamp_ns = self.previous_stamp_ns
        self.previous_stamp_ns = stamp_ns
        if previous_stamp_ns is None:
            return 1

        elapsed_ns = stamp_ns - previous_stamp_ns
        if not 0 < elapsed_ns <= self.record_jump_ns:
            self.get_logger().info('Record jump detected, resetting detector')
            self.detector.reset()
            return 1
        return max(1, round(elapsed_ns / self.lidar_period_ns))

    def obstacle_markers(self, header, result):
        markers = MarkerArray(markers=[Marker(header=header, action=Marker.DELETEALL)])
        if not result.detections:
            return markers

        frame_transform = result.debug['frame_transform']
        rotation = np.asarray(frame_transform['rotation'])
        translation = np.asarray(frame_transform['translation_m'])
        qx, qy, qz, qw = Rotation.from_matrix(rotation.T).as_quat()

        for index, detection in enumerate(result.detections):
            box_min = np.asarray(detection.bbox_min_xyz)
            box_max = np.asarray(detection.bbox_max_xyz)
            center = ((box_min + box_max) / 2 - translation) @ rotation
            size = np.maximum(box_max - box_min, MIN_BOX_SIZE_M)

            marker = Marker(header=header, ns='obstacle', id=index, type=Marker.CUBE, color=BOX_COLOR)
            marker.pose.position = Point(x=center[0], y=center[1], z=center[2])
            marker.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            marker.scale = Vector3(x=size[0], y=size[1], z=size[2])
            markers.markers.append(marker)
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
