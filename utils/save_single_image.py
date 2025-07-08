#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

# --- 配置 ---
# 要订阅的图像话题
IMAGE_TOPIC = '/camera/color/image_raw' 
# 保存的图片文件名
OUTPUT_FILENAME = 'test_image.jpg'

class SingleImageSaver(Node):
    """
    该节点订阅一个图像话题，保存接收到的第一张图像，然后关闭。
    """
    def __init__(self, output_path):
        super().__init__('single_image_saver_node')
        
        self.bridge = CvBridge()
        self.image_saved = False
        self.output_path = output_path
        
        # 创建订阅者
        # QoS Profile 的深度设置为 1 即可，因为我们只需要一张图
        self.subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            1)
        
        self.get_logger().info(f"等待来自 '{IMAGE_TOPIC}' 话题的图像...")
        self.get_logger().info(f"图像将被保存到: {os.path.abspath(self.output_path)}")

    def image_callback(self, msg):
        """
        接收到图像消息时的回调函数。
        """
        # 如果已经保存过图像，则直接返回，不做任何事
        if self.image_saved:
            return

        self.get_logger().info("成功接收到图像!")

        try:
            # 使用 cv_bridge 将 ROS 图像消息转换为 OpenCV 格式 (BGR8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge 转换失败: {e}")
            return

        # 保存图像
        try:
            cv2.imwrite(self.output_path, cv_image)
            self.get_logger().info(f"图像已成功保存到: {self.output_path}")
            self.image_saved = True # 设置标志位，防止再次保存
        except Exception as e:
            self.get_logger().error(f"保存图像失败: {e}")

        # 任务完成，关闭 ROS 节点
        self.get_logger().info("任务完成，正在关闭节点...")
        # 调用 rclpy.shutdown() 会让 main 函数中的 rclpy.spin() 退出
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    
    # 检查图像话题是否存在
    # (这是一个可选的辅助步骤，但对用户很友好)
    temp_node = rclpy.create_node('topic_checker')
    topics = temp_node.get_topic_names_and_types()
    if not any(topic_name == IMAGE_TOPIC for topic_name, _ in topics):
        print(f"\n[错误] 话题 '{IMAGE_TOPIC}' 当前不活跃。")
        print("请确保你的相机驱动节点正在运行并发布图像。")
        print("你可以使用 'ros2 topic list' 命令来检查可用的话题。\n")
        temp_node.destroy_node()
        rclpy.shutdown()
        return
    temp_node.destroy_node()
    
    try:
        image_saver_node = SingleImageSaver(OUTPUT_FILENAME)
        # 启动节点并等待回调函数完成任务
        # 当回调函数中调用 rclpy.shutdown() 时，spin 会停止
        rclpy.spin(image_saver_node)
    except KeyboardInterrupt:
        print("程序被用户中断。")
    finally:
        # 清理资源
        if 'image_saver_node' in locals() and image_saver_node.handle:
            image_saver_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

