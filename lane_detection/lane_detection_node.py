"""ROS2 node running the lane finding pipeline on the raw camera stream."""

import numpy as np
import rclpy
from car_msgs.msg import CameraExtrinsics
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64

from .perspective import DEFAULT_ROI_FAR, DEFAULT_ROI_NEAR, DEFAULT_ROI_WIDTH, PerspectiveTransform
from .pipeline import LaneFindingPipeline
from .visualization import draw_birds_eye

# Best effort matches both reliable and best effort camera publishers.
IMAGE_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)

PARAMETERS = [
    ('image_topic', '/zed/left_camera/image_raw'),
    ('camera_info_topic', '/my/camera_info'),
    ('extrinsics_topic', '/my/camera_extrinsics'),
    ('output_image_topic', '/lane_detection/image'),
    ('birds_eye_topic', '/lane_detection/birds_eye'),
    ('curve_radius_topic', '/lane_detection/curve_radius'),
    ('center_offset_topic', '/lane_detection/center_offset'),
    ('roi_width', DEFAULT_ROI_WIDTH),
    ('roi_near', DEFAULT_ROI_NEAR),
    ('roi_far', DEFAULT_ROI_FAR),
]


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection')
        self.declare_parameters(namespace='', parameters=PARAMETERS)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_size = None
        self.rotation_matrix = None
        self.translation = None
        self.pipeline = None

        self.image_pub = self.create_publisher(Image, self.get_parameter('output_image_topic').value, 1)
        self.birds_eye_pub = self.create_publisher(Image, self.get_parameter('birds_eye_topic').value, 1)
        self.curve_radius_pub = self.create_publisher(Float64, self.get_parameter('curve_radius_topic').value, 10)
        self.center_offset_pub = self.create_publisher(Float64, self.get_parameter('center_offset_topic').value, 10)

        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.camera_info_callback, 10)
        # Volatile also matches non-latched publishers (e.g. bag playback); the vehicle stack
        # republishes the extrinsics every second.
        self.create_subscription(
            CameraExtrinsics, self.get_parameter('extrinsics_topic').value, self.extrinsics_callback, 1)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.image_callback, IMAGE_QOS)

        self.get_logger().info(
            'Lane detection parameters:\n' + '\n'.join(
                f'  {name}: {self.get_parameter(name).value}' for name, _ in PARAMETERS
            )
        )
        self.get_logger().info('Waiting for camera info and extrinsics...')

    def camera_info_callback(self, msg):
        camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.array(msg.d, dtype=np.float64)
        size = (int(msg.width), int(msg.height))
        if (self.camera_matrix is not None and size == self.camera_info_size
                and np.array_equal(camera_matrix, self.camera_matrix)
                and np.array_equal(dist_coeffs, self.dist_coeffs)):
            return
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_info_size = size
        self.pipeline = None
        self.get_logger().info(
            f'Camera info received ({size[0]}x{size[1]}):\nK=\n{camera_matrix}\nD={dist_coeffs}')

    def extrinsics_callback(self, msg):
        rotation_matrix = np.array(msg.rotation_matrix, dtype=np.float64).reshape(3, 3)
        translation = np.array(msg.translation, dtype=np.float64)
        if (self.rotation_matrix is not None and np.array_equal(rotation_matrix, self.rotation_matrix)
                and np.array_equal(translation, self.translation)):
            return
        self.rotation_matrix = rotation_matrix
        self.translation = translation
        self.pipeline = None
        self.get_logger().info(f'Camera extrinsics received:\nR=\n{rotation_matrix}\nt={translation}')

    def setup_pipeline(self, img_size):
        if all(self.camera_info_size) and img_size != self.camera_info_size:
            self.get_logger().warn(
                f'Image is {img_size[0]}x{img_size[1]} but the camera info is for '
                f'{self.camera_info_size[0]}x{self.camera_info_size[1]}; the lane geometry will be wrong.'
            )
        try:
            perspective = PerspectiveTransform(
                img_size,
                self.camera_matrix,
                self.dist_coeffs,
                self.rotation_matrix,
                self.translation,
                float(self.get_parameter('roi_width').value),
                float(self.get_parameter('roi_near').value),
                float(self.get_parameter('roi_far').value),
            )
        except ValueError as exc:
            self.get_logger().error(f'Cannot set up the perspective transform: {exc}', throttle_duration_sec=5.0)
            return False
        self.pipeline = LaneFindingPipeline(perspective)
        self.get_logger().info(
            f'Perspective transform ready for {img_size[0]}x{img_size[1]}; source points (px):\n{perspective.src}')
        return True

    def image_callback(self, msg):
        if self.camera_matrix is None or self.rotation_matrix is None:
            self.get_logger().warn('Waiting for camera info and extrinsics...', throttle_duration_sec=5.0)
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'Cannot convert image: {exc}', throttle_duration_sec=5.0)
            return

        img_size = (img.shape[1], img.shape[0])
        if self.pipeline is None or self.pipeline.perspective.img_size != img_size:
            if not self.setup_pipeline(img_size):
                return

        birds_eye, _ = self.pipeline.perspective.warp(img)
        try:
            detection = self.pipeline.detect(img)
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            # np.polyfit raises when a lane line has no pixels
            self.get_logger().warn(f'Lane lines not found: {exc}', throttle_duration_sec=2.0)
            self.publish_image(self.birds_eye_pub, birds_eye, msg.header)
            return

        self.publish_image(self.image_pub, detection.overlay, msg.header)
        self.publish_image(
            self.birds_eye_pub,
            draw_birds_eye(birds_eye, detection.ploty, detection.left_fitx, detection.right_fitx),
            msg.header,
        )
        self.curve_radius_pub.publish(Float64(data=float((detection.left_curverad + detection.right_curverad) / 2)))
        self.center_offset_pub.publish(Float64(data=float(detection.veh_pos)))

    def publish_image(self, publisher, img, header):
        out_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out_msg.header = header
        publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
