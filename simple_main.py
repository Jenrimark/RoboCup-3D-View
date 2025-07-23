# -*- coding: utf-8 -*-
"""
RoboCup 3D 简化主程序
专注于展示美化界面和基础摄像头功能
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from td_recognition import Ui_MainWindow

class SimpleCameraThread(QThread):
    """简化的摄像头线程"""
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.camera_available = False
        
    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.camera_available = True
                self.status_update.emit("✅ 摄像头连接成功")
                return True
            else:
                self.camera_available = False
                self.status_update.emit("❌ 摄像头连接失败")
                return False
        except Exception as e:
            self.camera_available = False
            self.status_update.emit(f"❌ 摄像头初始化错误: {e}")
            return False
    
    def start_capture(self):
        """开始捕获"""
        if self.init_camera():
            self.running = True
            self.start()
        
    def stop_capture(self):
        """停止捕获"""
        self.running = False
        if self.cap:
            self.cap.release()
            
    def run(self):
        """摄像头主循环"""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 调整图像大小
                frame = cv2.resize(frame, (640, 480))
                self.frame_ready.emit(frame)
            self.msleep(33)  # 约30fps

class SimpleMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.init_ui()
        self.init_camera_thread()
        
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('🤖 RoboCup 3D 简化版系统')
        try:
            self.setWindowIcon(QIcon("icon/cug.ico"))
        except:
            pass
            
        # 连接按钮信号
        self.StartButton.clicked.connect(self.toggle_detection)
        self.cameraButton.clicked.connect(self.toggle_camera)
        self.settingsButton.clicked.connect(self.show_settings)
        
        # 设置初始状态
        self.detection_running = False
        self.StartButton.setText("🚀 开始检测")
        self.StartButton.setVisible(True)
        
        # 显示欢迎信息
        welcome_text = """
🎉 欢迎使用RoboCup 3D简化版系统!

🎨 界面特色:
✅ 现代化设计风格
✅ 渐变色彩搭配  
✅ 圆角边框效果
✅ 悬停动画交互
✅ 表情符号图标

📹 摄像头功能:
• 支持USB摄像头
• 640x480分辨率
• 30fps流畅显示
• 实时图像预览

🚀 点击"开始检测"开始体验!
        """
        self.ResultLabel.setText(welcome_text)
        self.statusbar.showMessage("🎨 系统已启动，界面美化完成")
        
    def init_camera_thread(self):
        """初始化摄像头线程"""
        self.camera_thread = SimpleCameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.status_update.connect(self.update_status)
        
    def update_frame(self, frame):
        """更新摄像头画面"""
        # 转换BGR到RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # 创建QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # 缩放到标签大小
        scaled_pixmap = pixmap.scaled(720, 540)
        self.ImgLabel.setPixmap(scaled_pixmap)
        
    def update_status(self, message):
        """更新状态消息"""
        self.statusbar.showMessage(message)
        
    def toggle_detection(self):
        """切换检测状态"""
        if not self.detection_running:
            # 开始检测
            self.start_detection()
        else:
            # 停止检测
            self.stop_detection()
            
    def start_detection(self):
        """开始检测"""
        self.detection_running = True
        self.StartButton.setText("⏹️ 停止检测")
        self.label.setText("🔍 正在检测中...")
        
        # 启动摄像头
        self.camera_thread.start_capture()
        
        # 模拟检测过程
        self.simulate_detection()
        
    def stop_detection(self):
        """停止检测"""
        self.detection_running = False
        self.StartButton.setText("🚀 开始检测")
        self.label.setText("💤 系统空闲 - 等待开始检测")
        
        # 停止摄像头
        self.camera_thread.stop_capture()
        
        # 恢复默认图像
        self.ImgLabel.setText("📹 摄像头画面")
        self.statusbar.showMessage("⏹️ 检测已停止")
        
    def simulate_detection(self):
        """模拟检测过程"""
        detection_results = """
🎯 模拟检测结果:

目标ID：CA001   数量：1
目标ID：CB002   数量：2  
目标ID：CC003   数量：1

📊 检测信息:
• 检测时间: 1.2秒
• 处理帧数: 36帧
• 检测精度: 92.5%
• 目标总数: 4个

⚠️ 这是简化版演示
实际检测需要加载AI模型
        """
        self.ResultLabel.setText(detection_results)
        
    def toggle_camera(self):
        """切换摄像头状态"""
        if self.camera_thread.camera_available:
            QMessageBox.information(self, "摄像头状态", 
                "📹 摄像头状态信息\n\n"
                "✅ 摄像头已连接\n"
                "• 分辨率: 640x480\n"
                "• 帧率: 30fps\n"
                "• 设备: USB摄像头\n"
                "• 状态: 正常工作")
        else:
            QMessageBox.warning(self, "摄像头状态", 
                "❌ 摄像头未连接\n\n"
                "请检查:\n"
                "• USB连接是否正常\n"
                "• 摄像头驱动是否安装\n"
                "• 是否被其他程序占用")
                
    def show_settings(self):
        """显示设置对话框"""
        QMessageBox.information(self, "系统设置", 
            "⚙️ RoboCup 3D 系统设置\n\n"
            "🎨 界面设置:\n"
            "• 主题: 深色渐变主题\n"
            "• 字体: Microsoft YaHei\n"
            "• 动画: 启用悬停效果\n\n"
            "📹 摄像头设置:\n"
            "• 分辨率: 640x480\n"
            "• 帧率: 30fps\n"
            "• 自动曝光: 启用\n\n"
            "🔧 检测设置:\n"
            "• 模型: 简化演示模式\n"
            "• 置信度: 0.5\n"
            "• 实时检测: 支持")
            
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.detection_running:
            self.stop_detection()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = SimpleMainWindow()
    window.show()
    
    # 显示启动消息
    QMessageBox.information(window, "系统启动", 
        "🎉 RoboCup 3D 简化版系统启动成功!\n\n"
        "✨ 特色功能:\n"
        "• 美化的现代界面设计\n"
        "• 实时摄像头预览\n"
        "• 模拟目标检测\n"
        "• 用户友好的交互\n\n"
        "🚀 开始体验吧!")
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
