    def image_callback(self, image_msg, info_msg):
        # 尝试获取锁，如果失败（意味着上一个回调还在处理），则直接返回以丢弃当前帧
        # 这是保证实时性的关键，避免因处理延迟导致消息堆积
        if not self.lock.acquire(blocking=False):
            self.get_logger().warn('Dropping a frame, inference is not fast enough for the input rate.')
            return

        try:
            # --- 计时器初始化 ---
            t0 = self.get_clock().now()

            # [第1部分] 将ROS消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            original_shape = cv_image.shape
            t1 = self.get_clock().now()

            # [第2部分] 执行推理 (这个函数内部已经有打印，但我们在这里测量包含预处理的总时间)
            raw_output = self.trt_model.infer(cv_image)
            t2 = self.get_clock().now()

            # [第3部分] 核心后处理 (重塑、缩放、应用参数)
            depth_map = raw_output.reshape(self.trt_model.output_shape[-2:])
            depth_resized = cv2.resize(depth_map, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            metric_depth = (depth_resized * self.param_scale + self.param_shift).astype(np.float32)
            t3 = self.get_clock().now()

            # [第4部分] 结果打包与发布 (包括可视化和消息转换)
            # 1. 准备并发布原始深度图 (32FC1)
            depth_msg = self.bridge.cv2_to_imgmsg(metric_depth, encoding='32FC1')
            depth_msg.header = image_msg.header
            self.depth_pub.publish(depth_msg)

            # 2. 准备并发布可视化深度图 (BGR8)
            depth_visual = cv2.normalize(metric_depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            colored_depth = cv2.applyColorMap(depth_visual, cv2.COLORMAP_INFERNO)
            depth_color_msg = self.bridge.cv2_to_imgmsg(colored_depth, encoding='bgr8')
            depth_color_msg.header = image_msg.header
            self.depth_color_pub.publish(depth_color_msg)

            # 3. 发布深度图对应的相机信息
            info_msg.header = image_msg.header
            self.depth_info_pub.publish(info_msg)
            t4 = self.get_clock().now()

            # --- 生成并打印详细的耗时报告 ---
            time_ros_to_cv = (t1 - t0).nanoseconds / 1e6
            time_inference_full = (t2 - t1).nanoseconds / 1e6
            time_post_processing = (t3 - t2).nanoseconds / 1e6
            time_publishing_and_viz = (t4 - t3).nanoseconds / 1e6
            time_total = (t4 - t0).nanoseconds / 1e6

            # 使用一个多行字符串来格式化日志，使其更易读
            log_message = (
                f"--- Timing Breakdown (ms) ---\n"
                f"  1. ROS->CV Convert:    {time_ros_to_cv:6.2f}\n"
                f"  2. Full Inference:     {time_inference_full:6.2f} (incl. preprocess)\n"
                f"  3. Post-Processing:    {time_post_processing:6.2f} (resize & scale)\n"
                f"  4. Viz & Publishing:   {time_publishing_and_viz:6.2f} (colorize, convert, pub)\n"
                f"--------------------------------\n"
                f"  Total Cycle Time:      {time_total:6.2f}"
            )
            self.get_logger().info(log_message)

        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
        finally:
            # 确保锁被释放
            self.lock.release()
