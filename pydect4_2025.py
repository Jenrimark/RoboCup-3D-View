import sys
import os
from td_recognition import Ui_MainWindow  # 加载我们的布局
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import pyqtSignal
import cv2
import argparse
import numpy as np
import pyorbbecsdk as orsdk
import torch
import shutil
from ultralytics import YOLO
import socket
import time

# --- Global Configuration ---
address = '172.27.246.124'  # 修改为您的电脑IP地址
GPU_DEVICE = False  # 禁用GPU，使用CPU模式
detect_time = [16, 16, 16]
SAT_NUM = 0.9

# --- Global Data Structures ---
number = [0] * 10
one_round = [0] * 10
myList = [([0] * 100) for _ in range(10)]
data = {i: f"C{chr(65 + i // 2)}00{i % 2 + 1}" for i in range(8)}
data.update({8: 'W001', 9: 'W002'})
rgb_dict = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255),
            6: (128, 0, 0), 7: (0, 128, 0), 8: (0, 0, 128), 9: (128, 128, 0)}
elseObject = [6, 7, 8, 9]

# --- Global Socket Object ---
# This object is now managed by the NetworkThread, making the global one less critical.
client = socket.socket()

# --- Helper Functions ---
def draw_detection_box(image, bbox, label, confidence, rgb):
    """Draws a bounding box and label on an image."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), rgb, 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
    return image

# --- Protocol Class ---
class Protocol:
    def __init__(self, bs=None):
        self.bs = bytearray(0)
    def add_int32(self, val): self.bs += val.to_bytes(4, byteorder='big')
    def add_str(self, val):
        bytes_val = val.encode('utf8')
        self.bs += len(bytes_val).to_bytes(4, byteorder='big')
        self.bs += bytes_val
    def pack_send(self, dtype, con):
        self.add_int32(dtype)
        self.add_str(con)

# --- Predictor Class (YOLO Segmentation) ---
class Predictor:
    def __init__(self, model_path):
        self.segment_model = YOLO(model_path)
    def predict(self, im0):
        results = self.segment_model(im0, device='cpu', conf=0.25, half=False)
        masks = np.zeros((480, 640), dtype=np.uint8)
        for r in results:
            if r.masks is None:
                masks[160:320, 220:420] = 1
                return masks
            mask_data, center = r.masks.data, 320
            min_distance, best_mask = float('inf'), None
            for i in range(len(r.masks)):
                dis = abs(center - np.mean(r.masks.xy[i][:, 0]))
                if dis < min_distance:
                    min_distance, best_mask = dis, mask_data[i]
            if best_mask is not None: return best_mask.numpy()
        return masks

# --- Threading Classes ---
# **REPLACED** class SocketConnectionThread with NetworkThread
class NetworkThread(QThread):
    """网络连接线程"""
    connection_ready = pyqtSignal(bool)
    status_updated = pyqtSignal(str)

    def __init__(self, host, port, timeout=5, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self._is_running = True
        self.parent_window = parent

    def run(self):
        try:
            self.status_updated.emit(f"正在连接服务器 {self.host}:{self.port}...")
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(self.timeout)
            self.client.connect((self.host, self.port))
            self.status_updated.emit("服务器连接成功")

            # Set the connection status in the camera thread
            if self.parent_window and hasattr(self.parent_window, 'camera_thread'):
                self.parent_window.camera_thread.set_connection_status(True, self.client)

            self.status_updated.emit("正在发送队名...")
            team_protocol = Protocol()
            # dataType=0 用于发送队名
            team_protocol.pack_send(0, "CUG-ZXPX")
            team_data_to_send = bytes(team_protocol.bs)
            self.client.sendall(team_data_to_send)
            self.status_updated.emit("队名已发送，网络准备就绪")

            self.connection_ready.emit(True)

            # 保持线程存活以接收发送数据的请求
            while self._is_running:
                self.msleep(100)  # Idle loop

        except Exception as e:
            if self.parent_window and hasattr(self.parent_window, 'camera_thread'):
                self.parent_window.camera_thread.set_connection_status(False, None)
            self.status_updated.emit(f"网络任务失败: {str(e)}")
            self.connection_ready.emit(False)
        finally:
            if self.client:
                self.client.close()
            print("网络线程已结束。")

    @pyqtSlot(bytes)
    def send_data(self, data):
        """发送数据到服务器"""
        if self.client and self._is_running:
            try:
                self.client.sendall(data)
                self.status_updated.emit("测量数据已发送")
                print("√ 结果已通过Socket发送。")
            except Exception as e:
                self.status_updated.emit(f"发送数据失败: {e}")
                print(f"× Socket发送失败: {e}")
                self.stop()  # 如果发送失败，可能连接已断开，停止线程
        else:
            self.status_updated.emit("无法发送：网络未连接或线程已停止")
            print("× 未连接到服务器，跳过发送。")


    def stop(self):
        self._is_running = False
        # Let the run loop exit naturally

class ModelLoaderThread(QThread):
    """Dedicated thread for loading ML models."""
    models_loaded = pyqtSignal(object, object)
    update_status = pyqtSignal(str)
    loading_error = pyqtSignal(str)

    def run(self):
        try:
            self.update_status.emit("正在加载主检测模型...")
            model = YOLO('weights/best.pt')
            print("√ 主检测模型加载成功")
            self.update_status.emit("正在加载分割模型...")
            predictor = Predictor('weights/yuan0517.pt')
            print("√ 分割模型加载成功")
            self.models_loaded.emit(model, predictor)
        except Exception as e:
            error_msg = f"× 模型加载失败: {e}"
            print(error_msg)
            self.loading_error.emit(error_msg)

class CameraThread(QThread):
    """Main thread for camera handling, detection, and data processing."""
    update_image = pyqtSignal(np.ndarray)
    update_label = pyqtSignal(str)
    update_result = pyqtSignal(str)
    detection_finished = pyqtSignal()

    def __init__(self, Window):
        super(CameraThread, self).__init__()
        self.window = Window
        self._is_running = True
        self.should_detect = False
        self.models_ready = False
        self.is_client_connected = False
        self.pipeline, self.cap, self.model, self.predictor = None, None, None, None
        
        out = opt.output
        if os.path.exists(out): shutil.rmtree(out)
        os.makedirs(out)

    def stop(self):
        self._is_running = False
        print("停止信号已发送至摄像头线程。")

    def set_models(self, model, predictor):
        self.model, self.predictor = model, predictor
        self.models_ready = True
        self.should_detect = True
        print("√ 模型已设置，检测准备就绪。")

    def set_connection_status(self, is_connected, sock_obj):
        global client
        self.is_client_connected = is_connected
        if is_connected:
            client = sock_obj # Keep the global client reference updated

    def initialize_camera(self):
        try:
            print("尝试连接Orbbec相机...")
            self.pipeline = orsdk.Pipeline()
            config = orsdk.Config()
            config.enable_stream(self.pipeline.get_stream_profile_list(orsdk.OBSensorType.COLOR_SENSOR).get_default_video_stream_profile())
            config.enable_stream(self.pipeline.get_stream_profile_list(orsdk.OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile())
            config.set_align_mode(orsdk.OBAlignMode.HW_MODE)
            self.pipeline.start(config)
            print("√ 成功连接Orbbec相机")
            return True
        except Exception as e:
            print(f"× Orbbec相机连接失败: {e}. 切换到电脑自带摄像头...")
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened(): raise IOError("无法打开电脑摄像头")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print("√ 成功连接电脑摄像头")
                return True
            except Exception as e_cam:
                if self._is_running: self.update_label.emit(f"× 错误：无法打开摄像头: {e_cam}")
                return False

    def _cleanup(self):
        print("正在清理摄像头线程资源...")
        if self.cap: self.cap.release()
        if self.pipeline: self.pipeline.stop()
        print("摄像头线程资源已释放。")

    def run(self):
        if not self.initialize_camera():
            self._cleanup()
            return

        max_times = 8
        times = 0

        while self._is_running:
            color_image = None
            if self.pipeline:
                frames = self.pipeline.wait_for_frames(100)
                if frames:
                    color_frame = frames.get_color_frame()
                    if color_frame: color_image = self.color_frame_to_bgr_img0(color_frame)
            elif self.cap:
                ret, frame = self.cap.read()
                if ret: color_image = cv2.resize(frame, (640, 480))

            if color_image is None:
                time.sleep(0.01)
                continue

            if self.models_ready and self.should_detect:
                img = color_image.copy()
                if times == 0: self.update_label.emit("开始检测...")
                
                results = self.model(img, device='cpu', conf=opt.conf_thres, half=False)
                masks = self.predictor.predict(img)
                one_round = [0] * 10
                
                if results[0] and results[0].boxes:
                    for i in range(len(results[0].boxes.cls)):
                        cls_index = int(results[0].boxes.cls[i])
                        if cls_index in elseObject or cls_index >= 10: continue
                        xyxy = results[0].boxes[i].xyxy.numpy()[0]
                        x_center, y_center = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                        if masks[y_center, x_center] > 0:
                            one_round[cls_index] += 1
                            img = draw_detection_box(img, xyxy, data[cls_index], results[0].boxes.conf[i], rgb_dict[cls_index])
                
                if self._is_running:
                    self.update_image.emit(img)
                    self.update_label.emit(f"检测中... ({times + 1}/{max_times + 1})")
                
                for k in range(10): myList[k][one_round[k]] += 1
                
                if times >= max_times:
                    for i in range(10):
                        k = 1 if myList[i][0] < (times + 1) * SAT_NUM else 0
                        for j in range(5, k - 1, -1):
                            if myList[i][j] > 1: number[i] = j; break
                    self.should_detect = False
                    self.process_detection_results()
                    while not self.should_detect and self._is_running: time.sleep(0.1)
                    times = 0
                    for i in range(10): number[i] = 0; myList[i] = [0] * 100
                    continue
                times += 1
            else:
                if self._is_running:
                    self.update_image.emit(color_image)
                    if not self.models_ready:
                        self.update_label.emit("正在加载模型... (实时预览中)")
                time.sleep(0.03)
        
        self._cleanup()

    def process_detection_results(self):
        if not self._is_running: return
        self.update_label.emit("√ 检测完成!")
        result_str = '\n'.join([f"目标ID：{data[i]}   数量：{n}" for i, n in enumerate(number) if n > 0])
        result_str2 = ''.join([f"Goal_ID={data[i]};Num={n}\n" for i, n in enumerate(number) if n > 0])
        data_to_send = 'START\n' + result_str2 + 'END'
        print(data_to_send)

        # **MODIFICATION**: Use the main window's signal to request data sending
        if self.is_client_connected:
            p = Protocol()
            p.pack_send(1, data_to_send)
            data_bytes = bytes(p.bs)
            self.window.request_send_data.emit(data_bytes)
        else:
            print("× 未连接到服务器，跳过发送。")

        try:
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_output")
            os.makedirs(result_dir, exist_ok=True)
            with open(os.path.join(result_dir, "CUG-CUG2.4G-R1.txt"), 'w+', encoding='utf-8') as f:
                f.write(data_to_send + '\n')
            print(f"√ 结果已保存到文件。")
        except Exception as e:
            print(f"× 保存结果文件失败: {e}")

        if self._is_running: self.update_result.emit(result_str); self.detection_finished.emit()

    def color_frame_to_bgr_img0(self, frame):
        data = np.asanyarray(frame.get_data())
        if frame.get_format() == orsdk.OBFormat.RGB:
            return cv2.cvtColor(np.resize(data, (480, 640, 3)), cv2.COLOR_RGB2BGR)
        elif frame.get_format() == orsdk.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        return None

# --- Main Window Class ---
class UsingTest(QMainWindow, Ui_MainWindow):
    # **NEW**: Signal to request network thread to send data
    request_send_data = pyqtSignal(bytes)
    
    def __init__(self, *args, **kwargs):
        super(UsingTest, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowTitle('RoboCup 3D识别 - v2025')
        self.setWindowIcon(QIcon("icon/cug.ico"))
        
        # Create all threads
        self.camera_thread = CameraThread(self)
        self.model_loader_thread = ModelLoaderThread()
        # **MODIFICATION**: Use the new NetworkThread
        self.network_thread = NetworkThread(host=address, port=6666, parent=self)
        
        # Connect signals to slots
        self.camera_thread.update_image.connect(self.update_camera_image)
        self.camera_thread.update_label.connect(self.update_status_label)
        self.camera_thread.update_result.connect(self.update_result_text)
        self.camera_thread.detection_finished.connect(self.on_detection_finished)
        
        self.model_loader_thread.models_loaded.connect(self.on_models_loaded)
        self.model_loader_thread.update_status.connect(self.update_status_label)
        self.model_loader_thread.loading_error.connect(self.on_loading_error)
        
        # **MODIFICATION**: Connect NetworkThread signals
        self.network_thread.connection_ready.connect(self.on_network_ready)
        self.network_thread.status_updated.connect(self.update_status_label)
        
        # **NEW**: Connect the request signal to the network thread's sending slot
        self.request_send_data.connect(self.network_thread.send_data)
        
        # UI setup
        self.StartButton.setText("重新检测")
        self.StartButton.setVisible(False)
        self.StartButton.clicked.connect(self.restart_detection)
        self.update_status_label("正在初始化...")
        
        # Start all threads
        self.network_thread.start()
        self.model_loader_thread.start()
        self.camera_thread.start()

    # **MODIFICATION**: New slot for network readiness
    @pyqtSlot(bool)
    def on_network_ready(self, success):
        # The status label is already updated by the thread itself.
        # This slot is mainly to confirm the network part of initialization is done.
        print(f"Network initialization complete. Success: {success}")

    @pyqtSlot(object, object)
    def on_models_loaded(self, model, predictor):
        self.update_status_label("√ 模型加载完成，即将开始自动检测...")
        if self.camera_thread.isRunning():
            self.camera_thread.set_models(model, predictor)

    @pyqtSlot(str)
    def on_loading_error(self, error_message):
        self.update_status_label(error_message)
        QMessageBox.critical(self, "错误", f"模型加载失败:\n{error_message}")

    @pyqtSlot(np.ndarray)
    def update_camera_image(self, img):
        h, w, ch = img.shape
        q_img = QImage(img.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.ImgLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ImgLabel.setPixmap(pixmap)

    @pyqtSlot(str)
    def update_status_label(self, text):
        self.label.setText(text)

    @pyqtSlot(str)
    def update_result_text(self, text):
        self.ResultLabel.setText(text)

    @pyqtSlot()
    def on_detection_finished(self):
        self.StartButton.setVisible(True)
        self.update_status_label("√ 检测完成！点击下方按钮重新检测")

    def restart_detection(self):
        self.StartButton.setVisible(False)
        self.ResultLabel.setText("")
        if self.camera_thread.models_ready and self.camera_thread.isRunning():
            self.camera_thread.should_detect = True
            self.update_status_label("重新开始检测...")
        else:
            self.update_status_label("无法开始检测：线程未运行或模型未加载。")
    
    def closeEvent(self, event):
        print("正在关闭应用程序...")
        # Stop all threads gracefully
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        if self.network_thread.isRunning():
            self.network_thread.stop()
            self.network_thread.wait()
        if self.model_loader_thread.isRunning():
            self.model_loader_thread.quit() # Model loader can be quit directly
            self.model_loader_thread.wait()
        print("所有线程已终止。窗口关闭。")
        event.accept()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    app = QApplication(sys.argv)
    win = UsingTest()
    win.show()
    sys.exit(app.exec_())