#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机切换测试脚本
测试Orbbec相机和普通摄像头的自动切换功能
"""

import cv2
import numpy as np
import time

try:
    import pyorbbecsdk as orsdk
    ORBBEC_AVAILABLE = True
except ImportError:
    print("pyorbbecsdk 未安装，将只使用普通摄像头")
    ORBBEC_AVAILABLE = False

class CameraManager:
    def __init__(self):
        self.use_orbbec = False
        self.pipeline = None
        self.cap = None
        self.initialize_camera()
    
    def initialize_camera(self):
        """初始化相机，优先使用Orbbec相机，失败则使用普通摄像头"""
        if ORBBEC_AVAILABLE:
            # 首先尝试初始化Orbbec相机
            try:
                print("尝试连接Orbbec相机...")
                self.pipeline = orsdk.Pipeline()
                
                # 配置Orbbec相机
                config = orsdk.Config()
                color_profile_list = self.pipeline.get_stream_profile_list(orsdk.OBSensorType.COLOR_SENSOR)
                color_profile = color_profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
                
                # 开始流式传输
                profile = self.pipeline.start(config)
                self.use_orbbec = True
                print("✓ 成功连接Orbbec相机")
                return True
                
            except Exception as e:
                print(f"✗ Orbbec相机连接失败: {e}")
                print("切换到电脑自带摄像头...")
        
        # 如果Orbbec相机失败或不可用，使用普通摄像头
        self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            print("✗ 错误：无法打开电脑摄像头")
            return False
        else:
            print("✓ 成功连接电脑摄像头")
            return True
    
    def get_frame(self):
        """获取一帧图像"""
        if self.use_orbbec:
            # 使用Orbbec相机获取图像
            try:
                frames = self.pipeline.wait_for_frames(100)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    return None
                
                # 转换为numpy数组
                color_image = self.color_frame_to_bgr_img(color_frame)
                return color_image
            except Exception as e:
                print(f"Orbbec相机读取失败: {e}")
                return None
        else:
            # 使用普通摄像头获取图像
            ret, color_image = self.cap.read()
            
            if not ret or color_image is None:
                return None
            
            # 调整图像大小到640x480
            color_image = cv2.resize(color_image, (640, 480))
            return color_image
    
    def color_frame_to_bgr_img(self, frame):
        """将Orbbec彩图数据帧转换为numpy格式的BGR彩图"""
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        
        new_width = 640
        new_height = 480
        
        if color_format == orsdk.OBFormat.RGB:
            image = np.resize(data, (new_height, new_width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == orsdk.OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            print("不支持彩图数据格式: {}".format(color_format))
            return None
        return image
    
    def release(self):
        """释放相机资源"""
        if self.cap is not None:
            self.cap.release()
            print("普通摄像头资源已释放")
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                print("Orbbec相机资源已释放")
            except:
                pass

def main():
    """主函数，测试相机功能"""
    print("=== 相机切换测试 ===")
    
    # 初始化相机管理器
    camera = CameraManager()
    
    print(f"\n当前使用的相机类型: {'Orbbec相机' if camera.use_orbbec else '普通摄像头'}")
    print("\n按 'q' 键退出，按 's' 键保存当前帧")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 获取一帧图像
            frame = camera.get_frame()
            
            if frame is None:
                print("无法获取图像帧")
                continue
            
            frame_count += 1
            
            # 在图像上添加信息
            camera_type = "Orbbec Camera" if camera.use_orbbec else "Web Camera"
            cv2.putText(frame, f"Camera: {camera_type}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 计算FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('Camera Test', frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"camera_test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"保存图像: {filename}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 清理资源
        camera.release()
        cv2.destroyAllWindows()
        print("测试结束")

if __name__ == "__main__":
    main()
