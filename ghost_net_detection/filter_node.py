import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from filterpy import discrete_bayes


class FilterNode(Node):
    def __init__(self):
        super().__init__('filter_node')

        self.current_belief = np.array([0.9, 0.1])

        # given prior that a net was detected
        likelihood_true_positive = 0.6
        likelihood_false_positive = 0.4
        assert likelihood_true_positive + likelihood_false_positive == 1.0
        self.likelihoods_detection = np.array([likelihood_false_positive, likelihood_true_positive])

        # given prior that a net was not detected
        likelihood_true_negative = 0.8
        likelihood_false_negative = 0.2
        assert likelihood_true_negative + likelihood_false_negative == 1.0
        self.likelihoods_no_detection = np.array([likelihood_true_negative, likelihood_false_negative])

        self.prediction_publisher = self.create_publisher(
            Float32,
            '/detection/prediction',
            10
        )

        self.detection_subscriber = self.create_subscription(
            Bool,
            '/detection/is_net',
            self.detection_callback,
            10
        )


    def detection_callback(self, msg: Bool):
        if msg.data:
            likelihood = self.likelihoods_detection
        else:
            likelihood = self.likelihoods_no_detection

        self.current_belief = discrete_bayes.update(likelihood, self.current_belief)

        self.prediction_publisher.publish(Float32(data=self.current_belief[1]))


def main(args=None):
    rclpy.init(args=args)

    filter_node = FilterNode()
    rclpy.spin(filter_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
