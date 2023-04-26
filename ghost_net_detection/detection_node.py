from math import pi, atan2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from ghost_net_detection.svm_model import load_svm_model


NUM_BINS = 8


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        self.model = load_svm_model('model.pkl')

        self.bridge = CvBridge()

        self.detection_publisher = self.create_publisher(
            Bool,
            '/detection/is_net',
            10
        )

        self.image_subscriber = self.create_subscription(
            Image,
            '/left/image_rect',
            self.image_callback,
            10
        )

        self.show_image = True

    def image_callback(self, msg: Image):
        if self.show_image:
            self.show_image = False
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            lines = find_lines(cv_image)
            angles = line_angles(lines)

            hist, bin_edges = np.histogram(angles, bins=NUM_BINS, range=(0, pi))

            is_net = self.predict_net(hist)

            self.detection_publisher.publish(Bool(data=is_net))

            # plot the histogram
            plt.clf()
            plt.title('Histogram of line angles')
            plt.xlabel('Angle (radians)')
            plt.ylabel('Count')
            plt.xlim(0, pi)
            plt.bar(
                bin_edges[:-1],
                hist,
                width=(bin_edges[1] - bin_edges[0]),
                align='edge',
                # orthogonal angle bins get the same color 
                color=[cm.get_cmap('hsv')(2*a / pi % 1) for a in bin_edges[:-1]]
            )
            plt.pause(0.001)
        self.show_image = True

    def predict_net(self, hist):
        predictions = self.model.predict([hist])
        return float(predictions[0]) > 0.5


def show_img(img, winname, win_row=0, win_col=0):
    cv2.imshow(winname, img)
    cv2.waitKey(1)

    win_x = win_col * img.shape[1]
    win_y = win_row * img.shape[0]
    cv2.moveWindow(winname, win_x, win_y)


def show_lines(img, lines, win_row=0, win_col=0):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    show_img(img, 'Lines', win_row, win_col)


def find_lines(img):
    # convert from rgb to grayscale if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur image to remove high frequency noise
    blured_img = cv2.GaussianBlur(img, (5, 5), 0)

    # equalize image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equilized_img = clahe.apply(blured_img)
    show_img(equilized_img, 'Contrast Limited Adaptive Histogram Equalization')

    # canny edge detection
    sigma = 0.33
    med_intensity = np.median(equilized_img)
    canny_threshold_lower = int(max(0, (1.0 - sigma) * med_intensity))
    canny_threshold_upper = int(min(255, (1.0 + sigma) * med_intensity))
    edge_img = cv2.Canny(
        image=equilized_img,
        threshold1=canny_threshold_lower,
        threshold2=canny_threshold_upper,
        edges=None,
        apertureSize=3,
    )
    show_img(edge_img, 'Edges')

    # find lines using hough transform
    lines = cv2.HoughLinesP(
            image=edge_img,
            rho=1,
            theta=pi/180,
            threshold=60,
            lines=None,
            minLineLength=20,
            maxLineGap=3
    )

    if lines is None:
        print('No lines found')
        return []

    show_lines(img, lines, win_row=1, win_col=0)

    return lines


def line_angles(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = atan2(y2 - y1, x2 - x1)
        angle %= pi
        angles.append(angle)
    return angles


def lines_centroid(lines):
    center_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        center_points.append(((x1 + x2) / 2, (y1 + y2) / 2))

    center_point = np.mean(center_points, axis=0)
    return center_point


def main(args=None):
    rclpy.init(args=args)

    detection_node = DetectionNode()
    rclpy.spin(detection_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
