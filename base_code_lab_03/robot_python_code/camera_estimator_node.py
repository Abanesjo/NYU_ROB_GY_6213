import cv2 
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation

from camera_estimator import CameraEstimator
import parameters

class CameraEstimatorNode(Node):
    def __init__(self, camera_id, marker_length, camera_matrix, dist_coeffs):
        super().__init__('camera_estimator_node')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/camera/odom", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image", 10)
        self.cv_bridge = CvBridge()
        self.estimator = CameraEstimator(camera_id, marker_length, camera_matrix, dist_coeffs)

    def publish_marker_tf(self, tvec: np.ndarray, rvec: np.ndarray, child_frame: str) -> None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw =  Rotation.from_matrix(rotation_matrix).as_quat()

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

    def publish_odom(self, tvec: np.ndarray, rvec: np.ndarray, child_frame: str) -> None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "camera_frame"
        odom.child_frame_id = child_frame
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

            cv2.imshow("aruco test", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            rclpy.spin_once(self, timeout_sec=0.0)

        self.estimator.close()


def main() -> None:
    rclpy.init()
    node = CameraEstimatorNode(parameters.camera_id, parameters.marker_length, parameters.camera_matrix, parameters.dist_coeffs)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



