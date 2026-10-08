import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        
        self.timer_period = 1.0 / 30.0  
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().error("Não foi possível abrir a câmera!")
            return

        self.bridge = CvBridge()
        self.get_logger().info("Nó Publicador de Câmera iniciado com sucesso publicando em '/camera/image_raw'.")

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_link'
            
            self.publisher_.publish(msg)
        else:
            self.get_logger().warn("Falha ao ler o frame da câmera.")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()