import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation

from camera_estimator import CameraEstimator
import parameters


class CameraEstimatorNode(Node):
    def __init__(self, camera_id, marker_length, camera_matrix, dist_coeffs):
        super().__init__("camera_estimator_node")
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/camera/odom", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image", 10)
        self.cv_bridge = CvBridge()
        self.estimator = CameraEstimator(
            camera_id, marker_length, camera_matrix, dist_coeffs
        )
        self.publish_static_transforms()

    def publish_static_transforms(self) -> None:
        stamp = self.get_clock().now().to_msg()

        odom_to_tripod = TransformStamped()
        odom_to_tripod.header.stamp = stamp
        odom_to_tripod.header.frame_id = "odom"
        odom_to_tripod.child_frame_id = "tripod"
        odom_to_tripod.transform.translation.x = float(parameters.tripod_x)
        odom_to_tripod.transform.translation.y = float(parameters.tripod_y)
        odom_to_tripod.transform.translation.z = float(parameters.tripod_z)
        tripod_quat = Rotation.from_euler(
            "xyz",
            [parameters.tripod_roll, parameters.tripod_pitch, parameters.tripod_yaw],
        ).as_quat()
        odom_to_tripod.transform.rotation.x = float(tripod_quat[0])
        odom_to_tripod.transform.rotation.y = float(tripod_quat[1])
        odom_to_tripod.transform.rotation.z = float(tripod_quat[2])
        odom_to_tripod.transform.rotation.w = float(tripod_quat[3])

        tripod_to_camera = TransformStamped()
        tripod_to_camera.header.stamp = stamp
        tripod_to_camera.header.frame_id = "tripod"
        tripod_to_camera.child_frame_id = "camera_frame"
        tripod_to_camera.transform.translation.x = 0.0
        tripod_to_camera.transform.translation.y = 0.0
        tripod_to_camera.transform.translation.z = 0.0
        camera_quat = Rotation.from_euler(
            "xyz",
            [parameters.camera_roll, parameters.camera_pitch, parameters.camera_yaw],
        ).as_quat()
        tripod_to_camera.transform.rotation.x = float(camera_quat[0])
        tripod_to_camera.transform.rotation.y = float(camera_quat[1])
        tripod_to_camera.transform.rotation.z = float(camera_quat[2])
        tripod_to_camera.transform.rotation.w = float(camera_quat[3])

        self.static_tf_broadcaster.sendTransform([odom_to_tripod, tripod_to_camera])

    def publish_marker_tf(
        self, tvec: np.ndarray, rvec: np.ndarray, child_frame: str
    ) -> None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "camera_frame"
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(tvec[0])
        transform.transform.translation.y = float(tvec[1])
        transform.transform.translation.z = float(tvec[2])
        transform.transform.rotation.x = float(qx)
        transform.transform.rotation.y = float(qy)
        transform.transform.rotation.z = float(qz)
        transform.transform.rotation.w = float(qw)

        self.tf_broadcaster.sendTransform(transform)

    def publish_odom(
        self, tvec: np.ndarray, rvec: np.ndarray, child_frame: str
    ) -> None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "camera_frame"
        odom.child_frame_id = child_frame

        # Odometry
        odom.pose.pose.position.x = float(tvec[0])
        odom.pose.pose.position.y = float(tvec[1])
        odom.pose.pose.position.z = float(tvec[2])
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)

        self.odom_pub.publish(odom)

    def publish_image(self, frame: np.ndarray) -> None:
        msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        self.image_pub.publish(msg)

    def run(self) -> None:
        while rclpy.ok():
            ret, frame = self.estimator.read_frame()
            if not ret or frame is None:
                break

            vis, markers = self.estimator.process_frame(frame)
            marker_count = len(markers)
            for marker_id, tvec, rvec in markers:
                if marker_count == 1:
                    child_frame = "aruco_frame"
                else:
                    child_frame = f"aruco_frame_{marker_id}"
                self.publish_marker_tf(tvec, rvec, child_frame)
                self.publish_odom(tvec, rvec, child_frame)

            self.publish_image(vis)

            # cv2.imshow("aruco test", vis)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

            rclpy.spin_once(self, timeout_sec=0.0)

        self.estimator.close()


def main() -> None:
    rclpy.init()
    node = CameraEstimatorNode(
        parameters.camera_id,
        parameters.marker_length,
        parameters.camera_matrix,
        parameters.dist_coeffs,
    )
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
