import os
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


class SaveImageNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        self.declare_parameter('save_path', '')
        self.save_path = self.get_parameter('save_path').get_parameter_value().string_value
        assert self.save_path is not '', 'save_path must be set as a parameter (e.g. ros2 run ghost_net_detection save_image_node --ros-args -p save_path:=images/positive)'
        self.save_path = Path(self.save_path)

        os.makedirs(self.save_path, exist_ok=True)

        self.images_saved = 0

        self.bridge = CvBridge()

        self.image_subscriber = self.create_subscription(
            Image,
            '/left/image_rect',
            self.image_callback,
            10
        )

    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        path = (self.save_path / f'{self.images_saved}.png').as_posix()
        cv2.imwrite(path, cv_image)
        self.get_logger().info(f'Image {self.images_saved} saved as {path}')
        self.images_saved += 1


def main(args=None):
    rclpy.init(args=args)

    save_image_node = SaveImageNode()
    rclpy.spin(save_image_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
