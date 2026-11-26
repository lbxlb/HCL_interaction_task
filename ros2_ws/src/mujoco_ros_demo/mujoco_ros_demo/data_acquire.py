#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
import mujoco as mj
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import numpy as np

class DataAcquisitionNode(Node):
    def __init__(self):
        super().__init__('data_acquisition_node')
        
        # 发布器定义（多种数据类型）
        self.object_pos_pub = self.create_publisher(PointStamped, '/perception/sim_object_pos', 10)
        self.joint_pos_pub = self.create_publisher(Float64MultiArray, '/perception/sim_joint_pos', 10)
        self.robot_pose_pub = self.create_publisher(PoseStamped, '/perception/sim_robot_pose', 10)
        
        # ========== MuJoCo初始化（无渲染） ==========
        xml_path = Path(get_package_share_directory('mujoco_ros_demo')) / 'config' / 'humanoid.xml'
        self.get_logger().info(f'Loading MuJoCo model from: {xml_path}')
        
        # 加载模型和数据
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        
        # 定时采集数据（10Hz）
        self.timer = self.create_timer(1/10, self.publish_sim_data)
        self.get_logger().info('Data acquisition node (MuJoCo) started successfully!')

    def publish_sim_data(self):
        # 运行仿真一步
        mj.mj_step(self.model, self.data)
        
        # 1. 发布物体位置（示例：第一个物体）
        if self.data.xpos.size > 0:
            pos_msg = PointStamped()
            pos_msg.header.stamp = self.get_clock().now().to_msg()
            pos_msg.header.frame_id = 'sim_world'
            pos_msg.point.x = float(self.data.xpos[0, 0])
            pos_msg.point.y = float(self.data.xpos[0, 1])
            pos_msg.point.z = float(self.data.xpos[0, 2])
            self.object_pos_pub.publish(pos_msg)
        
        # 2. 发布关节位置
        joint_msg = Float64MultiArray()
        joint_msg.data = self.data.qpos.tolist()  # 所有关节位置
        self.joint_pos_pub.publish(joint_msg)
        
        # 3. 发布机器人姿态（示例：第一个body）
        if self.data.xpos.size > 0 and self.data.xquat.size > 0:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'sim_world'
            # 位置
            pose_msg.pose.position.x = float(self.data.xpos[1, 0])
            pose_msg.pose.position.y = float(self.data.xpos[1, 1])
            pose_msg.pose.position.z = float(self.data.xpos[1, 2])
            # 四元数姿态
            pose_msg.pose.orientation.w = float(self.data.xquat[1, 0])
            pose_msg.pose.orientation.x = float(self.data.xquat[1, 1])
            pose_msg.pose.orientation.y = float(self.data.xquat[1, 2])
            pose_msg.pose.orientation.z = float(self.data.xquat[1, 3])
            self.robot_pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DataAcquisitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
