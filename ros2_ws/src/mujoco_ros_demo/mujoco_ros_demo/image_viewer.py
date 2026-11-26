# 在mujoco_ros_demo包中新建image_viewer.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.subscription = self.create_subscription(
            Image,
            '/perception/raw_image',
            self.image_callback,
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # 将ROS Image消息转为cv2图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Conversion error: {e}')
            return
        
        # 显示图像
        cv2.imshow('Camera Image', cv_image)
        cv2.waitKey(1)  # 必须加这行，否则窗口无响应

def main(args=None):
    rclpy.init(args=args)
    viewer = ImageViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        viewer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()