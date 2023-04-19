import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class TetherEncoderNode(Node):
    def __init__(self):
        super().__init__('tether_encoder_node')

        #Constants
        De = 0.05        #[m]      Outside diameter
        Di = 0.04        #[m]      Inside diameter
        pw = 1025        #[kg/m^3] Density water
        pi = 8830        #[kg/m^3] Density copper
        g = 9.81         #[m/s^2]

        pulse = 40
        self.deg_pulse = 360/pulse	
        dist_per_rev = 0.140	#	[m]
        self.radius = dist_per_rev/2
        self.deg_rad = 0.0174532925

        maxrangeofmotion = 45
        maxpulses_per_dir = 28  #0.7*40 (revolution*pulse_per_revolution)
        self.deg_per_pulse_a = maxrangeofmotion/(2*maxpulses_per_dir)

        #Variables (s and alfa obtained from encoders)
        #s = 3                   #value from measurment (Cable length, measured from encoder attatched to reel) np.arange(1,100,1)
        #a = 70*np.pi/180        #value from measurment (Feed angle, measured at buoy end) np.arange(1,100,1)
        #T = np.arange(1,100,1)  #value from measurment (Tension in cable, measured at buoy end)
        a = 0
        s = 0


        self.length_publisher = self.create_publisher(Float64, '/tether/length', 10)
        self.angle_publisher = self.create_publisher(Float64, '/tether/angle', 10)

    def listen_to_encoder(self):
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        length_msg = Float64()
        angle_msg = Float64()
        try:
            while True:
                # Get encoder values
                arduino_message = ser.readline()
                if arduino_message:
                    arduino_message_s = arduino_message.decode()
                    
                    if arduino_message_s[0] == 's':
                        s_cnt = int(arduino_message_s[1:])		
                        self.get_logger().info("Cable length count is: ", s_cnt)		# This is encoder turns
                        s = s_cnt*self.deg_pulse*self.deg_rad*self.radius
                        length_msg.data = s
                        self.length_publisher.publish(length_msg)
                        self.get_logger().info("Cable length is: ", s)

                    elif arduino_message_s[0] == 'a':
                        angle_cnt = int(arduino_message_s[1:])		# This is encoder turns, has to be transformed to degrees
                        self.get_logger().info("Feed angle count is: ", angle_cnt)
                        angle = angle_cnt*self.deg_per_pulse_a
                        angle_msg.data = angle
                        self.angle_publisher.publish(angle_msg)
                        self.get_logger().info("Feed angle is: ", angle)
                    else:
                        self.get_logger().warn("Wrong received message")
        finally:
            ser.close()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    tether_encoder_node = TetherEncoderNode()
    tether_encoder_node.listen_to_encoder()
    rclpy.spin(tether_encoder_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
