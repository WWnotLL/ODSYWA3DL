import array

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Quaternion, TransformStamped, Vector3
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from core.config import Config
from core.pipeline import ObstacleDetector
from obstacle_detector_msgs.msg import Obstacle, ObstacleState

NUMPY_TYPES = {
    PointField.INT8: '<i1', PointField.UINT8: '<u1',
    PointField.INT16: '<i2', PointField.UINT16: '<u2',
    PointField.INT32: '<i4', PointField.UINT32: '<u4',
    PointField.FLOAT32: '<f4', PointField.FLOAT64: '<f8',
}
NULLABLE_CORE_PARAMETERS = ('gauge.y_center_offset_m', 'gauge.axis_slope', 'detector.max_range_m')
TRUSTED_AXIS_STATES = ('measured', 'held')
PREVIEW_FIELDS = [PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
                  for index, name in enumerate(('x', 'y', 'z', 'intensity'))]
MIN_BOX_SIZE_M = 0.05
LINE_WIDTH_M = 0.05
OBSTACLE_COLOR = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.5)
AXIS_COLOR = ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0)
CORRIDOR_COLOR = ColorRGBA(r=0.2, g=0.9, b=0.3, a=1.0)


def pointcloud_to_arrays(msg):
    point_type = np.dtype({
        'names': [field.name for field in msg.fields],
        'formats': [NUMPY_TYPES[field.datatype] for field in msg.fields],
        'offsets': [field.offset for field in msg.fields],
        'itemsize': msg.point_step,
    })
    points = np.frombuffer(msg.data, dtype=point_type, count=msg.width * msg.height)
    xyz = np.stack([points['x'], points['y'], points['z']], axis=1)
    if 'ring' in point_type.names and 'timestamp' in point_type.names:
        return xyz, points['intensity'], points['ring'], points['timestamp']
    return xyz, points['intensity'], None, None


def preview_cloud(header, xyz, intensity, point_step):
    keep = np.flatnonzero(np.any(xyz != 0, axis=1))[::point_step]
    points = np.empty(keep.size, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])
    points['x'], points['y'], points['z'] = xyz[keep].T
    points['intensity'] = intensity[keep]
    return PointCloud2(
        header=header, height=1, width=keep.size, fields=PREVIEW_FIELDS, is_bigendian=False,
        point_step=16, row_step=16 * keep.size, data=array.array('B', points.tobytes()), is_dense=True)


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


def make_transform(header, child_frame, rotation, translation):
    qx, qy, qz, qw = Rotation.from_matrix(rotation).as_quat()
    transform = TransformStamped(header=header, child_frame_id=child_frame)
    transform.transform.translation = Vector3(x=translation[0], y=translation[1], z=translation[2])
    transform.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return transform


class ObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        defaults_path = get_package_share_directory('obstacle_detector') + '/config/obstacle_detector.yaml'
        with open(defaults_path, encoding='utf-8') as file:
            defaults = yaml.safe_load(file)['obstacle_detector']['ros__parameters']

        pointcloud_topics = self.declare_parameter('pointcloud_topics', defaults['pointcloud_topics']).value
        self.track_frame = self.declare_parameter('track_frame', defaults['track_frame']).value
        self.lidar_period_ns = int(self.declare_parameter('lidar_period_s', defaults['lidar_period_s']).value * 1e9)
        self.record_jump_ns = int(self.declare_parameter('record_jump_s', defaults['record_jump_s']).value * 1e9)
        self.required_range_m = self.declare_parameter('required_range_m', defaults['required_range_m']).value
        self.preview_point_step = self.declare_parameter('preview_point_step', defaults['preview_point_step']).value

        self.detector = ObstacleDetector(self.declare_core_config(defaults['core']))
        self.previous_stamp_ns = None

        self.state_pub = self.create_publisher(ObstacleState, '/obstacle/state', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/obstacle/markers', 10)
        self.preview_pub = self.create_publisher(PointCloud2, '/obstacle/cloud_preview', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        for topic in pointcloud_topics:
            self.create_subscription(PointCloud2, topic, self.handle_pointcloud,
                                     QoSProfile(history=HistoryPolicy.KEEP_ALL))

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

        debug = result.debug
        if debug['checked']:
            self.publish_track_frame(msg.header, debug['frame_transform'])
        self.state_pub.publish(self.obstacle_state(msg.header.stamp, result))
        self.markers_pub.publish(self.track_markers(msg.header.stamp, result))
        self.preview_pub.publish(preview_cloud(msg.header, xyz, intensity, self.preview_point_step))

    def publish_track_frame(self, header, frame_transform):
        rotation = np.asarray(frame_transform['rotation'])
        translation = np.asarray(frame_transform['translation_m'])
        self.tf_broadcaster.sendTransform(
            make_transform(header, self.track_frame, rotation.T, -rotation.T @ translation))

    def frame_gap_since_previous(self, stamp_ns):
        previous_stamp_ns = self.previous_stamp_ns
        self.previous_stamp_ns = stamp_ns
        if previous_stamp_ns is None:
            return 1

        elapsed_ns = stamp_ns - previous_stamp_ns
        if not 0 < elapsed_ns <= self.record_jump_ns:
            self.detector.reset()
            return 1
        return max(1, round(elapsed_ns / self.lidar_period_ns))

    def obstacle_state(self, stamp, result):
        debug = result.debug
        corridor = debug.get('corridor')
        checked_range = corridor['x_m'][1] if debug['checked'] and corridor else float('nan')
        distance = result.nearest_distance_m

        state = ObstacleState()
        state.header.stamp = stamp
        state.header.frame_id = self.track_frame
        state.stop = bool(result.obstacle_found or debug.get('verdict', {}).get('stop', False))
        state.obstacle_found = bool(result.obstacle_found)
        state.nearest_distance_m = float('nan') if distance is None else distance
        state.checked_range_m = checked_range
        state.state = self.path_state(debug, state.stop, checked_range)

        for detection in result.detections:
            box_min = np.asarray(detection.bbox_min_xyz)
            box_max = np.asarray(detection.bbox_max_xyz)
            center = (box_min + box_max) / 2
            size = box_max - box_min
            state.obstacles.append(Obstacle(
                distance_m=detection.distance_m,
                center=Point(x=center[0], y=center[1], z=center[2]),
                size=Vector3(x=size[0], y=size[1], z=size[2]),
                n_points=detection.n_points,
                score=detection.score,
            ))
        return state

    def path_state(self, debug, stop, checked_range):
        if stop:
            return ObstacleState.OBSTACLE
        if not debug['checked'] or debug['axis']['state'] not in TRUSTED_AXIS_STATES:
            return ObstacleState.UNCHECKED
        if np.isnan(checked_range) or checked_range < self.required_range_m:
            return ObstacleState.UNCHECKED
        return ObstacleState.CLEAR

    def track_markers(self, stamp, result):
        clear_all = Marker(action=Marker.DELETEALL)
        clear_all.header.frame_id = self.track_frame
        markers = MarkerArray(markers=[clear_all])

        for index, detection in enumerate(result.detections):
            box_min = np.asarray(detection.bbox_min_xyz)
            box_max = np.asarray(detection.bbox_max_xyz)
            center = (box_min + box_max) / 2
            size = np.maximum(box_max - box_min, MIN_BOX_SIZE_M)

            box = self.track_marker(stamp, 'obstacle', index, Marker.CUBE, OBSTACLE_COLOR)
            box.pose.position = Point(x=center[0], y=center[1], z=center[2])
            box.scale = Vector3(x=size[0], y=size[1], z=size[2])
            markers.markers.append(box)

        polyline = (result.debug.get('axis') or {}).get('polyline')
        corridor = result.debug.get('corridor')
        if polyline:
            axis_points = np.asarray(polyline['points_xy_m'])
            markers.markers.append(self.track_line(stamp, 'axis', axis_points, 0.0, AXIS_COLOR))

            if corridor:
                near, far = corridor['x_m']
                half_width = corridor['half_width_max_m']
                checked_points = axis_points[(axis_points[:, 0] >= near) & (axis_points[:, 0] <= far)]
                markers.markers.append(self.track_line(stamp, 'corridor_left', checked_points, half_width, CORRIDOR_COLOR))
                markers.markers.append(self.track_line(stamp, 'corridor_right', checked_points, -half_width, CORRIDOR_COLOR))
        return markers

    def track_line(self, stamp, namespace, points_xy, lateral_offset, color):
        line = self.track_marker(stamp, namespace, 0, Marker.LINE_STRIP, color)
        line.scale.x = LINE_WIDTH_M
        line.points = [Point(x=x, y=y + lateral_offset, z=0.0) for x, y in points_xy]
        return line

    def track_marker(self, stamp, namespace, marker_id, marker_type, color):
        marker = Marker(ns=namespace, id=marker_id, type=marker_type, color=color)
        marker.pose.orientation.w = 1.0
        marker.header.frame_id = self.track_frame
        marker.header.stamp = stamp
        return marker


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
