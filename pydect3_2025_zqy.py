import sys
import os
from td_recognition import Ui_MainWindow
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
import cv2
import argparse
import numpy as np
import pyorbbecsdk as orsdk
import torch
import torch.backends.cudnn as cudnn
import shutil
from ultralytics import YOLO
import socket
import time

# --- 全局配置 (保持不变) ---
address = '172.27.127.115'
GPU_DEVICE = torch.cuda.is_available() # 自动检测GPU

data = {0: 'CA001', 1: 'CA002', 2: 'CA003', 3: 'CA004', 4: 'CB001', 5: 'CB002', 6: 'CB003', 7: 'CB004',
        8: 'CC001', 9: 'CC002', 10: 'CC003', 11: 'CC004', 12: 'CD001', 13: 'CD002', 14: 'CD003', 15: 'CD004',
        16: 'W001', 17: 'W002', 18: 'W003', 19: 'W004'}
rgb_dict = {i: tuple(np.random.randint(50, 255, 3).tolist()) for i in range(20)}
elseObject = [12, 13, 14, 15, 17, 18, 19]
else_dict = {0: 12, 1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 19, 7: 18}

client = socket.socket()
is_client = False
SAT_NUM = 0.9

# --- 工具函数 (保持不变) ---
def check_socket_connection(host, port):
    global is_client, client
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2)
        client.connect((host, port))
        print(f"✅ Socket connection to {host}:{port} successful!")
        is_client = True
    except socket.error as e:
        print(f"❌ Socket connection to {host}:{port} failed: {e}")
        is_client = False

def draw_detection_box(image, bbox, label, confidence, rgb):
    bbox = [int(coord) for coord in bbox]
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), rgb, 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
    return image

class Protocol:
    def __init__(self, bs=None): self.bs = bytearray(0)
    def add_int32(self, val): self.bs += bytearray(val.to_bytes(4, byteorder='big'))
    def add_str(self, val):
        bytes_val = bytearray(val.encode(encoding='utf8'))
        self.bs += bytearray(len(bytes_val).to_bytes(4, byteorder='big')) + bytes_val
    def pack_send(self, dtype, con): self.add_int32(dtype); self.add_str(con)

class Predictor:
    def __init__(self, model_path, device):
        self.segment_model = YOLO(model_path)
        self.device = device
    def predict(self, im0):
        results = self.segment_model(im0, device=self.device, conf=0.25, iou=0.45, half=False, max_det=100, verbose=False)
        masks = np.zeros((480, 640), dtype=np.uint8)
        for r in results:
            if r.masks is None:
                masks[160:320, 220:420] = 1
                return torch.from_numpy(masks) if GPU_DEVICE else masks
            mask_data = r.masks.data
            min_distance = float('inf')
            best_mask = None
            center_x = 320
            for index in range(len(r.masks)):
                xy = r.masks.xy[index]
                dis = abs(center_x - np.mean(xy[:, 0]))
                if dis < min_distance:
                    min_distance = dis
                    best_mask = mask_data[index]
            if best_mask is not None:
                masks = best_mask.cpu().numpy().astype(np.uint8)
            return torch.from_numpy(masks) if GPU_DEVICE else masks

# --- 线程类 new_thread (重大修改) ---
class new_thread(QThread):
    update_image = pyqtSignal(np.ndarray)
    update_label = pyqtSignal(str)
    update_result = pyqtSignal(str)
    ready_to_detect = pyqtSignal()

    RESTART_DELAY = 5 # 秒

    def __init__(self):
        super(new_thread, self).__init__()
        # 【关键修改】线程控制与状态封装
        self.is_running = True
        self.should_detect = False
        self.times = 0
        
        # 【关键修改】将状态变量封装到类实例中
        self.number = [0] * 20
        self.myList = [([0] * 100) for _ in range(20)]
        self.one_round = [0] * 20
        
        self.pipeline = None
        self.cap = None
        self.use_orbbec = False
        
        self.model = None
        self.model_w = None
        self.predictor = None
        
    def initialize_camera(self):
        # ... (此方法保持不变，为简洁省略)
        try:
            print("尝试连接Orbbec相机...")
            self.pipeline = orsdk.Pipeline()
            config = orsdk.Config()
            color_profile = self.pipeline.get_stream_profile_list(orsdk.OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
            config.enable_stream(color_profile)
            depth_profile = self.pipeline.get_stream_profile_list(orsdk.OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            config.set_align_mode(orsdk.OBAlignMode.HW_MODE)
            self.pipeline.start(config)
            self.use_orbbec = True
            print("✅ 成功连接Orbbec相机")
        except Exception as e:
            print(f"❌ Orbbec相机连接失败: {e}\n🔄 切换到电脑自带摄像头...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if self.cap.isOpened(): print("✅ 成功连接电脑摄像头")
            else: print("❌ 错误：无法打开电脑摄像头"); self.update_label.emit("❌ 错误：无法打开摄像头！")
    
    def initialize_models(self):
        self.update_label.emit("🔄 正在加载YOLO模型...")
        cudnn.benchmark = True
        self.device = 'cuda' if GPU_DEVICE else 'cpu'
        self.half = self.device != 'cpu'

        try:
            self.model = YOLO('det300.pt')
            self.predictor = Predictor('yuan0517.pt', self.device)
            self.model_w = YOLO('fruit.pt')
            print("✅ 所有模型加载成功")
            self.ready_to_detect.emit()
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.update_label.emit(f"❌ 模型加载失败: {e}")

    def get_frame(self):
        if self.use_orbbec:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        w, h, fmt = color_frame.get_width(), color_frame.get_height(), color_frame.get_format()
                        data = np.asanyarray(color_frame.get_data())
                        if fmt == orsdk.OBFormat.RGB:
                            img = cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                        elif fmt == orsdk.OBFormat.MJPG:
                            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        else: return None
                        return cv2.resize(img, (640, 480))
            except Exception: return None
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret: return cv2.resize(frame, (640, 480))
        return None

    # 【关键修改】线程的主循环
    def run(self):
        self.initialize_camera()
        self.initialize_models()
        check_socket_connection(address, 6666)
        
        max_times = 15

        # 【关键修改】使用 self.is_running 控制循环
        while self.is_running:
            color_image = self.get_frame()
            if color_image is None:
                time.sleep(0.5)
                continue

            img_to_show = color_image.copy()

            if self.should_detect and self.model:
                if self.times == 0: # 新一轮检测开始
                    p_start = Protocol(); p_start.pack_send(0, 'CUG2.4G')
                    if is_client:
                        try: client.send(p_start.bs)
                        except Exception as e: print(f"Socket send error: {e}")
                
                self.update_label.emit(f"🔍 检测中... ({self.times + 1}/{max_times})")
                
                # --- 检测逻辑 ---
                with torch.no_grad():
                    results = self.model(img_to_show, device=self.device, conf=opt.conf_thres, iou=opt.iou_thres, half=self.half, verbose=False)
                    results_w = self.model_w(img_to_show, device=self.device, conf=opt.conf_thres, iou=opt.iou_thres, half=self.half, verbose=False)
                    masks = self.predictor.predict(img_to_show)
                    
                    self.one_round = [0] * 20
                    
                    # 主模型
                    for r in results:
                        for box in r.boxes:
                            cls_index = int(box.cls[0])
                            if cls_index in elseObject: continue
                            if cls_index == 16: cls_index = 7
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x_c, y_c = (x1+x2)//2, (y1+y2)//2
                            if masks[y_c, x_c] > 0:
                                self.one_round[cls_index] += 1
                                img_to_show = draw_detection_box(img_to_show, (x1,y1,x2,y2), data[cls_index], box.conf[0], rgb_dict[cls_index])
                    # 水果模型
                    for r in results_w:
                        for box in r.boxes:
                            cls_index = else_dict[int(box.cls[0])]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x_c, y_c = (x1+x2)//2, (y1+y2)//2
                            if masks[y_c, x_c] > 0:
                                self.one_round[cls_index] += 1
                                img_to_show = draw_detection_box(img_to_show, (x1,y1,x2,y2), data[cls_index], box.conf[0], rgb_dict[cls_index])
                
                for k in range(20):
                    self.myList[k][self.one_round[k]] += 1
                
                self.times += 1

                # --- 检测结束，准备下一轮 ---
                if self.times >= max_times:
                    self.should_detect = False # 暂停检测
                    self.process_detection_results()
                    self.update_label.emit(f"✅ 本轮完成，{self.RESTART_DELAY}秒后重启...")
                    
                    # 使用非阻塞的方式等待
                    for _ in range(self.RESTART_DELAY * 10):
                        if not self.is_running: break # 如果此时关闭窗口，立即退出
                        time.sleep(0.1)

                    if self.is_running:
                        self.start_detection_round() # 重启下一轮
            
            self.update_image.emit(img_to_show)
            QThread.msleep(30) # 替代 time.sleep，更适合Qt

        self.cleanup() # 线程结束前清理资源

    def start_detection_round(self):
        """重置状态，开始新一轮检测"""
        print("指令：开始新一轮检测。")
        self.update_result.emit("") # 清空UI上的结果
        self.times = 0
        self.number = [0] * 20
        self.myList = [([0] * 100) for _ in range(20)]
        self.should_detect = True

    def process_detection_results(self):
        for i in range(20):
            k = 1 if self.myList[i][0] < self.times * SAT_NUM else 0
            for j in range(5, k - 1, -1):
                if self.myList[i][j] > 1: self.number[i] = j; break

        pri, pri2 = [], []
        for aa in range(20):
            if self.number[aa] != 0:
                pri.append(f"目标ID：{data[aa]}   数量：{self.number[aa]}")
                pri2.append(f"Goal_ID={data[aa]};Num={self.number[aa]}")
        
        result_str = '\n'.join(pri)
        data_end = 'START\n' + '\n'.join(pri2) + '\nEND'
        print(f"--- 检测结果 ---\n{data_end}\n-----------------")
        
        p_end = Protocol(); p_end.pack_send(1, data_end)
        if is_client:
            try: client.send(p_end.bs)
            except Exception as e: print(f"Socket send error: {e}")
        
        try:
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_output")
            os.makedirs(result_dir, exist_ok=True)
            filename = os.path.join(result_dir, "CUG-CUG2.4G-R1.txt")
            with open(filename, 'w', encoding='utf-8') as f: f.write(data_end + '\n')
            print(f"✅ 结果已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存结果文件失败: {e}")

        self.update_result.emit(result_str)

    # 【关键修改】线程停止方法
    def stop(self):
        print("接收到停止信号，线程准备退出...")
        self.is_running = False

    def cleanup(self):
        print("正在清理线程资源...")
        if self.cap: self.cap.release()
        if self.pipeline: self.pipeline.stop()
        if is_client: client.close()
        print("线程资源清理完毕。")

# --- 主窗口类 UsingTest (重大修改) ---
class UsingTest(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(UsingTest, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowTitle('🤖 RoboCup 3D识别 - v2025.3 (Auto-Run & Stable)')
        self.setWindowIcon(QIcon("icon/cug.ico"))

        self.new_thread = new_thread()

        self.new_thread.update_image.connect(self.update_camera_image)
        self.new_thread.update_label.connect(self.update_status_label)
        self.new_thread.update_result.connect(self.update_result_text)
        self.new_thread.ready_to_detect.connect(self.on_ready_to_detect)
        
        self.StartButton.setVisible(False)
        self.update_status_label("🔄 正在初始化...")

        self.new_thread.start()

    def on_ready_to_detect(self):
        """模型加载完成，自动开始第一轮检测"""
        self.update_status_label("✅ 初始化完成，自动开始检测...")
        self.new_thread.start_detection_round()

    @pyqtSlot(np.ndarray)
    def update_camera_image(self, img):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ImgLabel.setPixmap(pixmap.scaled(self.ImgLabel.width(), self.ImgLabel.height(), Qt.KeepAspectRatio))
        except Exception as e:
            print(f"UI Error: {e}")

    @pyqtSlot(str)
    def update_status_label(self, text):
        self.label.setText(text)

    @pyqtSlot(str)
    def update_result_text(self, text):
        self.ResultLabel.setText(text)

    # 【关键修改】重写关闭事件
    def closeEvent(self, event):
        """关闭窗口时，安全地停止线程"""
        print("窗口关闭事件触发。")
        self.new_thread.stop()  # 发送停止信号
        self.new_thread.wait()  # 等待线程完全退出
        print("线程已退出，关闭应用程序。")
        event.accept()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='new_best.pt')
    parser.add_argument('--source', default='inference/images')
    parser.add_argument('--output', default='inference/output')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.1)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    opt = parser.parse_args()

    app = QApplication(sys.argv)
    win = UsingTest()
    win.show()
    sys.exit(app.exec_())