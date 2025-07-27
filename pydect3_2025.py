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
## import torch.backends.cudnn as cudnn
import shutil
from ultralytics import YOLO
import socket
import time

address = '172.27.246.124'  # 修改为您的电脑IP地址
GPU_DEVICE = False  # 禁用GPU，使用CPU模式

last_number = []
for i in range(10):
    last_number.append(0)
# 识别时间
detect_time = [16, 16, 16]

# #待转向中间的间隔时间
sleep_time = 9

paper = 0.01

ground_dis = 0.2

# 0 值阈值
SAT_NUM = 0.9

# 统计个数
# 单轮的物品个数
number = []
one_round = []
for i in range(10):
    number.append(0)
    one_round.append(0)
myList = [([0] * 100) for i in range(10)]

# 对于某物品 单独置信度阈值
radio = [[0.5, 0.25, 0.5, 0.25],
         [0.5, 0.5, 0.25, 0.75],
         [0.25, 0.75, 0.5, 0.5],
         [0.5, 0.5, 0.5, 0.25],
         [0.5, 0.5, 0.5, 0.4]]

data = {0: 'CA001', 1: 'CA002', 2: 'CB001', 3: 'CB002', 4: 'CC001', 5: 'CC002', 6: 'CD001', 7: 'CD002',
        8: 'W001', 9: 'W002'}
rgb_dict = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255),
            6: (128, 0, 0), 7: (0, 128, 0), 8: (0, 0, 128), 9: (128, 128, 0)}
elseObject = [6, 7, 8, 9]
## else_dict = {0: 6, 1: 7, 2: 8, 3: 9}
client = socket.socket()


def check_socket_connection(host, port):
    try:
        # 创建一个 socket 对象
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 尝试连接到目标主机和端口
        client.connect((host, port))
        # 如果连接成功，输出提示信息
        print(f"Socket connection to {host}:{port} successful!")
        # 关闭 socket 连接
        # client.close()
        return True
    except socket.error as e:
        # 如果连接失败，输出连接错误信息
        print(f"Socket connection to {host}:{port} failed: {e}")
        return False


is_client = False


def adaptive_histogram_equalization_color(image_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    try:
        # 转换图像为 Lab 色彩空间
        lab_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2Lab)

        # 分离通道
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # 创建自适应直方图均衡化器
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # 对亮度通道应用自适应直方图均衡化
        l_channel_equalized = clahe.apply(l_channel)

        # 合并通道
        equalized_lab_image = cv2.merge([l_channel_equalized, a_channel, b_channel])

        # 转换回 BGR 色彩空间
        equalized_bgr_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_Lab2BGR)

        # 返回处理后的图像
        return equalized_bgr_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return image_np


def draw_detection_box(image, bbox, label, confidence, rgb):
    bbox = [int(coord) for coord in bbox]
    x1, y1, x2, y2 = bbox

    # 绘制检测框
    cv2.rectangle(image, (x1, y1), (x2, y2), rgb, 2)

    # 在检测框上方绘制文本信息
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)

    return image


class Protocol:

    def __init__(self, bs=None):
        self.bs = bytearray(0)

    def add_int32(self, val):  # 添加dataType（0：开始计时 1：发送识别结果 2：发送测量结果）
        bytes_val = bytearray(val.to_bytes(4, byteorder='big'))
        self.bs += bytes_val

    def add_str(self, val):  # 添加需要发送的字符串data(字符长度计算已包含在内)
        bytes_val = bytearray(val.encode(encoding='utf8'))
        bytes_length = bytearray(len(bytes_val).to_bytes(4, byteorder='big'))
        self.bs += (bytes_length + bytes_val)

    def pack_send(self, dtype, con):
        self.add_int32(dtype)
        self.add_str(con)


class Predictor:
    def __init__(self, model_path, device, conf_thres=0.25, iou_thres=0.45, max_det=100):
        self.segment_model = YOLO(model_path)
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def predict(self, im0, times):
        imgsz = (640, 480)
        results = self.segment_model(im0, device='cpu', conf=0.25, iou=self.iou_thres, half=False,
                                     max_det=self.max_det)
        max_conf = -1
        masks = [[0 for _ in range(640)] for _ in range(480)]
        masks = np.array(masks)

        for r in results:
            if r.masks is None:
                masks = [[0 for _ in range(640)] for _ in range(480)]
                masks = np.array(masks)
                masks[160:320, 220:420] = 1
                if GPU_DEVICE:
                    masks = torch.from_numpy(masks)
                return masks
            mask_data = r.masks.data
            confs = r.boxes.conf
            # for index in range(0, len(confs)):
            #     conf = confs[index]
            #     data = mask_data[index]
            #     if conf > max_conf:
            #         max_conf = conf
            #         if GPU_DEVICE:
            #             masks = data.cpu().numpy()
            #             masks = torch.from_numpy(masks)##.cuda()
            #         else:
            #             masks = data.numpy()
            center = 320
            min_distance = 640
            for index in range(len(r.masks)):
                xy = r.masks.xy[index]
                dis = abs(center - np.mean(xy[:, 0]))
                data = mask_data[index]
                if dis < min_distance:
                    # max_conf = conf
                    min_distance = dis
                    if GPU_DEVICE:
                        masks = data.cpu().numpy()
                        masks = torch.from_numpy(masks)##.cuda()
                    else:
                        masks = data.numpy()
            return masks


class detect_Flag_thread(QThread):

    def __init__(self, window):
        super(detect_Flag_thread, self).__init__()
        self.window = window

    def run(self):
        time.sleep(detect_time[self.window.round])
        self.window.detect_Flag = False


class new_thread(QThread):
    # 定义信号用于线程间通信
    update_image = pyqtSignal(np.ndarray)  # 更新图像信号
    update_label = pyqtSignal(str)  # 更新标签信号
    update_result = pyqtSignal(str)  # 更新结果信号
    show_start_button = pyqtSignal(bool)  # 显示开始按钮信号
    detection_finished = pyqtSignal()  # 检测完成信号

    def __init__(self, Window):
        super(new_thread, self).__init__()
        self.window = Window
        self.round = 0
        self.f = True
        self.use_orbbec = False  # 标记是否使用Orbbec相机
        self.pipeline = None
        self.cap = None
        self.should_detect = True  # 控制是否应该开始检测

        # 首先尝试初始化Orbbec相机
        try:
            print("尝试连接Orbbec相机...")
            self.pipeline = orsdk.Pipeline()

            # 配置Orbbec相机
            config = orsdk.Config()
            color_profile_list = self.pipeline.get_stream_profile_list(orsdk.OBSensorType.COLOR_SENSOR)
            color_profile = color_profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            depth_profile_list = self.pipeline.get_stream_profile_list(orsdk.OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            config.set_align_mode(orsdk.OBAlignMode.HW_MODE)

            # 开始流式传输
            profile = self.pipeline.start(config)
            # 修改相机内参
            config.set_align_mode(orsdk.OBAlignMode.HW_MODE)
            self.use_orbbec = True
            print("√ 成功连接Orbbec相机")

        except Exception as e:
            print(f"× Orbbec相机连接失败: {e}")
            print("切换到电脑自带摄像头...")

            # 如果Orbbec相机失败，使用普通摄像头
            self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                print("× 错误：无法打开电脑摄像头")
                return
            else:
                print("√ 成功连接电脑摄像头")

        ## cudnn.benchmark = True

        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # 初始化相关参数
        self.device = 'cpu'  # 强制使用CPU设备
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        self.half = False  # 强制使用CPU模式，禁用半精度

        # 加载yolo模型
        try:
            print("正在加载YOLO模型...")
            self.model = YOLO('best.pt')
            print("√ 主检测模型 best.pt 加载成功")

            # Get names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
            print(f"模型类别数: {len(self.names) if self.names else 'Unknown'}")

            self.predictor = Predictor('yuan0517.pt', 'cpu')
            print("√ 分割模型 yuan0517.pt 加载成功")

            ## self.model_w = YOLO('fruit.pt')
            ## print("√ 水果检测模型 fruit.pt 加载成功")

        except Exception as e:
            print(f"× 模型加载失败: {e}")
            self.model = None
            ## self.model_w = None

    def run(self):
        max_times = 15
        times = 0
        seconds_per = 1
        print('show camera scene....')
        is_client = check_socket_connection(address, 6666)

        # 发射信号通知UI模型加载完成，准备开始检测
        self.update_label.emit("√ 模型加载完成，准备开始检测...")
        self.show_start_button.emit(False)  # 隐藏开始按钮

        # 等待一秒让用户看到状态
        time.sleep(1)

        # 直接开始检测
        self.should_detect = True
        self.window.thread_run = False  # 设置为检测模式

        while True:
            if self.use_orbbec:
                # 使用Orbbec相机获取图像
                frames = self.pipeline.wait_for_frames(100)
                camera_param = self.pipeline.get_camera_param()

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # 将图像转换为numpy数组类型
                color_image = self.color_frame_to_bgr_img0(color_frame)
            else:
                # 使用普通摄像头获取图像
                ret, color_image = self.cap.read()

                if not ret or color_image is None:
                    print("无法读取摄像头帧")
                    continue

                # 调整图像大小到640x480
                color_image = cv2.resize(color_image, (640, 480))

            img_cpy = color_image.copy()
            img = color_image.copy()
            img0 = img

            # 如果需要检测，则进行检测
            if self.should_detect:
                with torch.no_grad():
                    # 预测，返回的结果包括识别的物体类别，框的位置，和该类物品的概率
                    if times % seconds_per != 0:
                        times += 1
                        continue
                    # 使用普通检测代替跟踪 (避免lap依赖问题)
                    print(f"开始YOLO检测，置信度阈值: {opt.conf_thres}")

                    # 检查模型是否加载成功
                    if self.model is None:
                        print("× 主检测模型未加载")
                        continue
                    ## if self.model_w is None:
                    ##     print("× 水果检测模型未加载")
                    ##     continue

                    results = self.model(img, augment=opt.augment, device='cpu',
                                        agnostic_nms=opt.agnostic_nms,
                                        classes=opt.classes, conf=opt.conf_thres, iou=opt.iou_thres,
                                        half=False)
                    ## results_w = self.model_w(img, augment=opt.augment, device=opt.device, half=self.half,
                    ##                        agnostic_nms=opt.agnostic_nms, classes=opt.classes,
                    ##                        conf=opt.conf_thres,
                    ##                        iou=opt.iou_thres)

                    result = results[0]
                    ## result_w = results_w[0]

                    # 调试信息
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        print(f"主模型检测到 {len(result.boxes)} 个目标")
                        if len(result.boxes) > 0:
                            print(f"   置信度范围: {result.boxes.conf.min():.3f} - {result.boxes.conf.max():.3f}")
                    else:
                        print("× 主模型未检测到任何目标")

                    ## if hasattr(result_w, 'boxes') and result_w.boxes is not None:
                    ##     print(f"水果模型检测到 {len(result_w.boxes)} 个目标")
                    ## else:
                    ##     print("× 水果模型未检测到任何目标")
                if times == 0:
                    self.detect_Flag = True
                    self.detect_new_thread = detect_Flag_thread(self)
                    self.detect_new_thread.start()
                    p_start = Protocol()
                    p_start.pack_send(0, 'CUG2.4G')
                    if is_client:
                        client.send(p_start.bs)

                masks = self.predictor.predict(img0, times)
                one_round.clear()
                for i in range(10):
                    one_round.append(0)
                if result is not None or len(result) != 0:
                    for index in range(len(result.boxes.cls)):
                        cls_index = int(result.boxes.cls[index])
                        if cls_index in elseObject:
                            continue
                        if GPU_DEVICE:
                            xyxy = result.boxes[index].xyxy.cpu().numpy()[0]
                            xyxy = torch.from_numpy(xyxy)##.cuda()
                        else:
                            xyxy = result.boxes[index].xyxy.numpy()[0]
                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        x_center = int((x1 + x2) / 2)
                        y_center = int(0.5 * y2 + 0.5 * y1)

                        if len(masks) > 0:
                            if masks[y_center][x_center] > 0:
                                item = cls_index
                                one_round[item] += 1
                                img = draw_detection_box(img, xyxy, data[item], result.boxes.conf[index],
                                                         rgb_dict[item])
                ## elif result_w is not None or len(result_w) != 0:
                ##     for index in range(len(result_w.boxes.cls)):
                ##         cls_index = else_dict[int(result_w.boxes.cls[index])]
                ##         if GPU_DEVICE:
                ##             xyxy = result_w.boxes[index].xyxy.cpu().numpy()[0]
                ##             xyxy = torch.from_numpy(xyxy)##.cuda()
                ##         else:
                ##             xyxy = result_w.boxes[index].xyxy.numpy()[0]
                ##         x1 = int(xyxy[0])
                ##         y1 = int(xyxy[1])
                ##         x2 = int(xyxy[2])
                ##         y2 = int(xyxy[3])
                ##         x_center = int((x1 + x2) / 2)
                ##         y_center = int(0.5 * y2 + 0.5 * y1)
                ##
                ##         if len(masks) > 0:
                ##             if masks[y_center][x_center] > 0:
                ##                 item = cls_index
                ##                 one_round[item] += 1
                ##                 img = draw_detection_box(img, xyxy, data[item], result_w.boxes.conf[index],
                ##                                          rgb_dict[item])
                else:
                    times += 1
                    continue

                # 发射信号更新UI
                self.update_image.emit(img)
                self.update_label.emit("正在检测中...")

                for k in range(10):
                    myList[k][one_round[k]] = myList[k][one_round[k]] + 1

                if times == max_times:
                    for i in range(10):
                        max_ = 0
                        k = 0
                        if myList[i][0] < times / seconds_per * SAT_NUM:
                            k = 1
                        for j in range(5, k - 1, -1):
                            # print(i, j, myList[i][j])
                            if 0 < myList[i][j] and myList[i][j] > 1:
                                number[i] = j
                                break

                if times >= max_times:
                    # 检测完成，停止检测并显示结果
                    self.should_detect = False
                    self.process_detection_results(is_client)

                    # 等待重新检测指令
                    while not self.should_detect:
                        # 显示最后一帧图像
                        self.update_image.emit(img)
                        time.sleep(0.1)

                    # 重置检测参数
                    times = 0
                    for i in range(10):
                        number[i] = 0
                        for j in range(100):
                            myList[i][j] = 0
                    continue
                times += 1
            else:
                # 如果不需要检测，只显示摄像头画面
                self.update_image.emit(color_image)
                self.update_label.emit("摄像头预览")
                time.sleep(0.03)  # 控制帧率

    def process_detection_results(self, is_client):
        """处理检测结果"""
        # 发射信号更新UI
        self.update_label.emit("√ 检测完成!")

        pri = []
        pri2 = []
        result_str = ''
        result_str2 = ''
        for aa in range(10):
            if number[aa] != 0:
                st = "目标ID：" + str(data[aa]) + "   数量：" + str(number[aa])
                pri.append(st)
                st = "Goal_ID=" + str(data[aa]) + ";Num=" + str(number[aa])
                pri2.append(st)

        for aa in range(len(pri)):
            result_str += pri[aa]
            result_str2 += pri2[aa]
            result_str += '\n'
            result_str2 += '\n'

        data_end = 'START\n' + result_str2 + 'END'
        print(data_end)
        p_end = Protocol()
        p_end.pack_send(1, data_end)
        if is_client:
            client.send(p_end.bs)

        # 创建结果输出目录和文件 (跨平台兼容)
        try:
            # 获取当前脚本目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result_dir = os.path.join(script_dir, "result_output")

            # 创建输出目录
            os.makedirs(result_dir, exist_ok=True)

            # 生成文件名
            filename = os.path.join(result_dir, "CUG-CUG2.4G-R1.txt")

            # 写入结果文件
            with open(filename, 'w+', encoding='utf-8') as f:
                f.write('START' + '\n')
                f.write(result_str2)
                f.write('END' + os.linesep + '\n')

            print(f"√ 结果已保存到: {filename}")
        except Exception as e:
            print(f"× 保存结果文件失败: {e}")

        # 发射信号更新结果显示和显示重新检测按钮
        self.update_result.emit(result_str)
        self.detection_finished.emit()

    def __del__(self):
        """析构函数，释放摄像头资源"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            print("普通摄像头资源已释放")
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            try:
                self.pipeline.stop()
                print("Orbbec相机资源已释放")
            except:
                pass

    def color_frame_to_bgr_img0(self, frame):
        '''将彩图数据帧转换为numpy格式的BGR彩图'''
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        image = np.zeros((height, width, 3), dtype=np.uint8)
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

    def depth_frame_to_numpy(self, depth_frame):
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        new_width = 640
        new_height = 480
        depth_data = depth_data.reshape((new_height, new_width))
        depth_data = depth_data.astype(np.float32) * scale
        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return depth_image


class UsingTest(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(UsingTest, self).__init__(*args, **kwargs)
        self.setupUi(self)  # 初始化u
        self._translate = QtCore.QCoreApplication.translate
        self.setWindowTitle('RoboCup 3D识别 - v2025')
        self.setWindowIcon(QIcon("icon/cug.ico"))
        self.thread_run = True
        self.new_thread = new_thread(self)

        # 连接信号槽
        self.new_thread.update_image.connect(self.update_camera_image)
        self.new_thread.update_label.connect(self.update_status_label)
        self.new_thread.update_result.connect(self.update_result_text)
        self.new_thread.show_start_button.connect(self.StartButton.setVisible)
        self.new_thread.detection_finished.connect(self.on_detection_finished)

        # 修改按钮文本为"重新检测"并隐藏
        self.StartButton.setText("重新检测")
        self.StartButton.setVisible(False)

        self.new_thread.start()
        self.StartButton.clicked.connect(self.restart_detection)

    def update_camera_image(self, img):
        """更新摄像头图像显示"""
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_display = QImage(img_display, img_display.shape[1], img_display.shape[0],
                           img_display.shape[1] * 3, QImage.Format_RGB888)
        img_display = QtGui.QPixmap(img_display).scaled(640, 480)
        self.ImgLabel.setPixmap(img_display)

    def update_status_label(self, text):
        """更新状态标签"""
        self.label.setText(text)

    def update_result_text(self, text):
        """更新结果显示"""
        self.ResultLabel.setText(text)

    def on_detection_finished(self):
        """检测完成后的处理"""
        self.StartButton.setVisible(True)  # 显示重新检测按钮
        self.update_status_label("√ 检测完成！点击下方按钮重新检测")

    def restart_detection(self):
        """重新开始检测"""
        self.StartButton.setVisible(False)  # 隐藏重新检测按钮
        self.ResultLabel.setText("")  # 清空结果显示
        self.new_thread.should_detect = True  # 设置检测标志
        self.update_status_label("重新开始检测...")

    def OpenImage(self):
        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap('inference\\images\\WIN_20201004_19_48_41_Pro.jpg')
        # self.label.setPixmap(jpg)
        self.ImgLabel.setPixmap(jpg)
        self.ResultLabel.setText('hello world!')
        self.label.setText(self._translate("MainWindow",
                                           "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">检测中</span></p></body></html>"))

    # letterbox变换  用于不改变原图像纵横比进行resize
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def detect(self):
        """原来的检测方法，现在已经不需要了"""
        pass

    def StartButton_clicked(self):
        """原来的按钮点击方法，现在重定向到重新检测"""
        self.restart_detection()


if __name__ == '__main__':  # 程序的入口

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
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
