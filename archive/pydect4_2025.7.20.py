import random
import sys
import os
from td_recognition import Ui_MainWindow  # 加载我们的布局
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QTimer
import cv2
import argparse
import numpy as np
import pyorbbecsdk as orsdk
import torch
import torch.backends.cudnn as cudnn
import shutil
import time
from ultralytics import YOLO
import socket

address = '192.168.0.113'  # 修改为您的电脑IP地址
GPU_DEVICE = True  # 禁用GPU，使用CPU模式

last_number = []
for i in range(20):
    last_number.append(0)
# 识别时间
detect_time = [14, 14, 14]

# #待转向中间的间隔时间
sleep_time = 4

paper = 0.01

ground_dis = 0.2

SAT_NUM = 0.9
# 误判斜率
xielv = 1
# 轮数 这里只能取 2，4


# 删除平面中离群点的阈值
num_p = 120

# 删除拟合平面的阈值
num_m = 101
# 统计个数
# 单轮的物品个数
number = []
one_round = []
for i in range(20):
    number.append(0)
    one_round.append(0)
myList = [([0] * 100) for i in range(20)]

# 物品基准置信度
base_conf = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.25, 5: 0.4,
            6: 0.2, 7: 0.6, 8: 0.5, 9: 0.7, 10: 0.3, 11:  0.25,
            12: 0.2, 13: 0.2, 14: 0.2, 15: 0.2, 16: 0.5, 17: 0.5,
            18: 0.5, 19: 0.5}
# 物品期望数量
expect_num = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
            6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11:  1,
            12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1,
            18: 1, 19: 1}
conf_shift = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
            6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11:  0,
            12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0,
            18: 0, 19: 0}
# 置信度阈值
conf_limit = [0.2, 0.7]
# conf_shift = 0.00
# conf_step = 0.05

data = {0: 'CA001', 1: 'CA002', 2: 'CA003', 3: 'CA004', 4: 'CB001', 5: 'CB002', 6: 'CB003', 7: 'CB004',
        8: 'CC001', 9: 'CC002', 10: 'CC003', 11: 'CC004', 12: 'CD001', 13: 'CD002', 14: 'CD003', 15: 'CD004',
        16: 'W001', 17: 'W002', 18: 'W003', 19: 'W004'}
rgb_dict = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255),
            6: (128, 0, 0), 7: (0, 128, 0), 8: (0, 0, 128), 9: (128, 128, 0), 10: (128, 0, 128), 11: (0, 128, 128),
            12: (64, 0, 0), 13: (0, 64, 0), 14: (0, 0, 64), 15: (64, 64, 0), 16: (64, 0, 64), 17: (0, 64, 64),
            18: (192, 192, 192), 19: (128, 128, 128)}
elseObject = [12, 13, 14, 15]
else_dict = {0: 12, 1: 13, 2: 14, 3: 15}
client = socket.socket()



is_client = False


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


class new_thread_(QThread):

    def __init__(self, Window):
        super(new_thread_, self).__init__()
        self.window = Window

    def run(self):
        time.sleep(sleep_time)
        self.window.thread_run = False


class Predictor:
    def __init__(self, model_path, device, conf_thres=0.25, iou_thres=0.45, max_det=100):
        # Load model
        # self.segment_model = YOLO(model_path, device=device, dnn=False, data='', fp16=False)
        self.segment_model = YOLO(model_path)
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def predict(self, im0, times):
        imgsz = (640, 480)
        results = self.segment_model(im0, device=self.device, conf=self.conf_thres, iou=self.iou_thres, half=False,
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
                    masks = torch.from_numpy(masks).cuda()
                return masks
            mask_data = r.masks.data
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
                        im0 = r.orig_img
                        masks = data.cpu().numpy()
                        if times <= 2:
                            output = 'mask_out' + str(times) + '.jpg'
                            mask = masks.astype(np.uint8)
                            masked_image = cv2.bitwise_and(im0, im0, mask=mask)
                            cv2.imwrite(output, masked_image)
                        masks = torch.from_numpy(masks).cuda()
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

    def __init__(self, Window):
        super(new_thread, self).__init__()
        self.window = Window
        self.round = 0
        self.f = True
        self.use_orbbec = False  # 标记是否使用Orbbec相机
        self.pipeline = None
        self.cap = None

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
            self.use_orbbec = True
            print("成功连接Orbbec相机")

        except Exception as e:
            print(f"Orbbec相机连接失败: {e}")
            print("切换到电脑自带摄像头...")

            # 如果Orbbec相机失败，使用普通摄像头
            self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                print("错误：无法打开电脑摄像头")
                return
            else:
                print("成功连接电脑摄像头")

        cudnn.benchmark = True

        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # 初始化相关参数
        # set_logging()
        # 获取脚本所在目录，确保模型文件路径正确
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.device = opt.device
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        self.half = self.device != 'cpu'  # half precision only supported on CUDA

        # 初始化模型
        self.init_model()

        # Get names and colors
        if self.model is not None:
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        else:
            self.names = {}
            self.colors = []

        predictor_path = os.path.join(script_dir, 'final0518seg.pt')
        self.predictor = Predictor(predictor_path, self.device)
        self.predictor2 = Predictor(predictor_path, self.device)

        fruit_model_path = os.path.join(script_dir, 'fruit.pt')
        self.model_w = YOLO(fruit_model_path)

    def init_model(self):
        """初始化YOLO模型和相关参数"""
        try:
            # 获取脚本所在目录，确保模型文件路径正确
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # 设置输出目录
            out = 'output'
            if os.path.exists(out):
                shutil.rmtree(out)
            os.makedirs(out, exist_ok=True)

            # 初始化设备参数
            self.device = 'cuda' if GPU_DEVICE and torch.cuda.is_available() else 'cpu'

            # 加载yolo模型
            model_path = os.path.join(script_dir, 'best518s.pt')
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print("✅ 主YOLO模型加载成功!")
            else:
                # 尝试加载其他可用的模型
                model_files = ['best.pt', 'yolov5s.pt']
                self.model = None
                for model_file in model_files:
                    full_path = os.path.join(script_dir, model_file)
                    if os.path.exists(full_path):
                        self.model = YOLO(full_path)
                        print(f"✅ 备用模型 {model_file} 加载成功!")
                        break
                if self.model is None:
                    print("❌ 未找到可用的YOLO模型文件")
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            self.model = None

    def run(self):
        global conf_shift
        times = 0
        seconds_per = 2
        # conf_shift = 0.00
        conf_step = 0.05
        start = time.time()
        print('show camera scene....')
        is_client = check_socket_connection(address, 6666)
        # color_image2 = cv2.imread("333.jpg")

        # 获取脚本所在目录
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        p_image_path = os.path.join(script_dir, 'P_image_51.jpg')
        masks = self.predictor.predict(p_image_path, 3)
        masks = self.predictor2.predict(p_image_path, 4)
        while True:
            if self.use_orbbec:
                # 使用Orbbec相机获取图像
                frames = self.pipeline.wait_for_frames(100)
                color_frame = frames.get_color_frame()
                # depth_frame = frames.get_depth_frame()

                if not color_frame:
                    continue
                # 将图像转换为numpy数组类型
                # depth_image = self.depth_frame_to_numpy(depth_frame)
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
            # depth_image_cpy = depth_image.copy()
            img = color_image.copy()
            # img0 = np.expand_dims(depth_image_cpy, axis=-1)
            # img0 = np.repeat(img0, 3, axis=-1)
            img0 = img
            if self.window.thread_run:

                # depth_image = np.asanyarray(aligned_depth_frame.get_data())

                img_display = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)
                img_display = QImage(img_display, img_display.shape[1], img_display.shape[0], img_display.shape[1] * 3,
                                     QImage.Format_RGB888)

                img_display = QtGui.QPixmap(img_display).scaled(640, 480)
                self.window.ImgLabel.setPixmap(img_display)
                self.window.label.setText("💤 系统空闲 - 等待开始检测")
                pred = self.model(img, augment=opt.augment, device=opt.device)[0]
                pred = self.model_w(img, augment=opt.augment, device=opt.device)[0]
                self.window.StartButton.setVisible(True)

            else:
                with torch.no_grad():
                    # 预测，返回的结果包括识别的物体类别，框的位置，和该类物品的概率
                    if times % seconds_per != 0:
                        times += 1
                        continue
                    if self.model is not None:
                        results = self.model(img, conf=0.1, device=self.device)
                    else:
                        results = []

                    if hasattr(self, 'model_w') and self.model_w is not None:
                        results_w = self.model_w(img, conf=0.1, device=self.device)
                    else:
                        results_w = []

                    result = results[0]
                    result_w = results_w[0]
                if times == 0:
                    # start = time.time()
                    self.window.TurningImg.setVisible(False)
                    self.detect_Flag = True
                    self.detect_new_thread = detect_Flag_thread(self)
                    self.detect_new_thread.start()
                    if self.round == 0:
                        p_start = Protocol()
                        p_start.pack_send(0, 'CUG2.4G')
                        if is_client:
                            client.send(p_start.bs)
                    if self.round < 2:
                        masks = self.predictor2.predict(img0, self.round)
                    else:
                        masks = self.predictor.predict(img0, self.round)

                self.window.ResultLabel.setText("")
                one_round.clear()
                for i in range(20):
                    one_round.append(0)
                flag1 = False
                flag2 = False
                if result is not None or len(result) != 0:
                    flag1 = True
                    for index in range(len(result.boxes.cls)):
                        cls_index = int(result.boxes.cls[index])
                        if cls_index in elseObject:
                            continue
                        #if cls_index == 16:
                            #cls_index = 7
                        if result.boxes.conf[index] < (base_conf[cls_index] + conf_shift[cls_index]):
                            continue
                        if GPU_DEVICE:
                            xyxy = result.boxes[index].xyxy.cpu().numpy()[0]
                            xyxy = torch.from_numpy(xyxy).cuda()
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
                if result_w is not None or len(result_w) != 0:
                    flag2 =True
                    for index in range(len(result_w.boxes.cls)):
                        cls_index = else_dict[int(result_w.boxes.cls[index])]
                        if result_w.boxes.conf[index] < (base_conf[cls_index] + conf_shift[cls_index]):
                            continue
                        if GPU_DEVICE:
                            xyxy = result_w.boxes[index].xyxy.cpu().numpy()[0]
                            xyxy = torch.from_numpy(xyxy).cuda()
                        else:
                            xyxy = result_w.boxes[index].xyxy.numpy()[0]
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
                                img = draw_detection_box(img, xyxy, data[item], result_w.boxes.conf[index],
                                                         rgb_dict[item])
                if not flag1 and not flag2:
                    times += 1
                    continue

                # 在平面上画出物品
                img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_display = QImage(img_display, img_display.shape[1], img_display.shape[0], img_display.shape[1] * 3,
                                     QImage.Format_RGB888)
                img_display = QtGui.QPixmap(img_display).scaled(640, 480)
                self.window.ImgLabel.setPixmap(img_display)
                if not GPU_DEVICE:
                    output_path = 'mask_out/mask' + str(times) + '.jpg'
                    mask = masks.astype(np.uint8)
                    masked_image = cv2.bitwise_and(img, img, mask=mask)
                    cv2.imwrite(output_path, masked_image)
                self.window.label.setText("🔍 正在检测中...")
                for k in range(20):
                    if one_round[k] > expect_num[k]:
                        t = (one_round[k] - expect_num[k]) / expect_num[k]
                        ra = random.random()
                        if ra < t:
                            conf_shift[k] += conf_step
                        if (base_conf[k] + conf_shift[k]) > conf_limit[1]:
                            conf_shift[k] = conf_limit[1] - base_conf[k]
                    if one_round[k] < expect_num[k]:
                        t = -(one_round[k] - expect_num[k]) / expect_num[k]
                        ra = random.random()
                        if ra < t:
                            conf_shift[k] -= conf_step
                        if (base_conf[k] + conf_shift[k]) < conf_limit[0]:
                            conf_shift[k] = conf_limit[0] + base_conf[k]
                for k in range(20):
                    myList[k][one_round[k]] = myList[k][one_round[k]] + 1
                times += 1

                if self.detect_Flag == False:
                    for i in range(20):
                        max_ = 0
                        k = 0
                        if self.round < 2:
                            if myList[i][0] < int(times / seconds_per) - 3:
                                k = 1
                        else:
                            if myList[i][0] < int(times / seconds_per) - 20:
                                k = 1
                        for j in range(5, k-1,-1):
                            print(i, j, myList[i][j])
                            if 0 < myList[i][j] and myList[i][j] > times / seconds_per * 0.1:
                                number[i] = j
                                break
                    for i in range(20):
                        last_number[i] = last_number[i] + number[i]
                        if last_number[i] > 5:
                            last_number[i] = 5;
                    if self.round < 2:
                        p_shift = Protocol()
                        p_shift.pack_send(3, "0000")
                        if is_client:
                            client.send(p_shift.bs)
                        self.window.TurningImg.setVisible(True)
                        self.window.label.setText("🔄 准备转向...")
                        self.window.thread_run = True
                        new_thread = new_thread_(self.window)
                        new_thread.start()

                    else:
                        self.window.label.setText("✅ 检测完成!")
                    pri = []
                    result_str = ''
                    for aa in range(20):
                        if number[aa] != 0:
                            # print("number[aa]",number[aa])
                            st = "目标ID：" + str(data[aa]) + "   数量：" + str(number[aa])
                            # print("st",st)
                            pri.append(st)
                    for aa in range(len(pri)):
                        result_str += pri[aa]
                        result_str += '\n'

                    self.window.ResultLabel.setText(result_str)
                    # print(result_str)

                    self.round = self.round + 1
                    times = 0
                    if self.round == 3:
                        break

        pri = []
        pri2 = []
        result_str = ''
        result_str2 = ''
        for aa in range(20):

            if last_number[aa] != 0:
                st = "目标ID：" + str(data[aa]) + "   数量：" + str(last_number[aa])
                pri.append(st)
                st = "Goal_ID=" + str(data[aa]) + ";Num=" + str(last_number[aa])
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
            filename = os.path.join(result_dir, "CUG-CUG2.4G-R2.txt")

            # 写入结果文件
            with open(filename, 'w+', encoding='utf-8') as f:
                f.write('START' + '\n')
                f.write(result_str2)
                f.write('END' + os.linesep + '\n')

            print(f"✅ 结果已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存结果文件失败: {e}")

        self.window.ResultLabel.setText(result_str)

        # cli
        # self.window.ResultLabel.repaint()

    def color_frame_to_bgr_img0(self, frame):
        '''将彩图数据帧转换为numpy格式的BGR彩图'''
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


class UsingTest(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(UsingTest, self).__init__(*args, **kwargs)
        self.setupUi(self)  # 初始化u
        self._translate = QtCore.QCoreApplication.translate
        self.setWindowTitle('🤖 RoboCup 3D识别 第 2 轮 - v2025')
        try:
            self.setWindowIcon(QIcon("icon/cug.ico"))
        except:
            pass
        self.thread_run = True
        self.new_thread = new_thread(self)
        self.new_thread.start()
        self.StartButton.clicked.connect(self.detect)

        # 连接新增按钮的信号 (按钮已隐藏)
        # self.cameraButton.clicked.connect(self.toggle_camera)
        # self.settingsButton.clicked.connect(self.show_settings)

        # 延迟启动相机测试，确保相机线程完全初始化
        self.camera_test_timer = QTimer()
        self.camera_test_timer.timeout.connect(self.auto_start_camera_test)
        self.camera_test_timer.setSingleShot(True)
        self.camera_test_timer.start(1000)  # 1秒后启动相机测试

    def auto_start_camera_test(self):
        """自动启动相机测试，显示实时画面"""
        try:
            # 设置状态提示
            self.statusbar.showMessage("🔍 正在自动检测相机设备...")
            self.label.setText("📹 正在启动相机预览...")

            # 等待相机线程完全初始化
            if not hasattr(self.new_thread, 'use_orbbec'):
                # 如果相机线程还没有初始化完成，再等待一下
                self.camera_test_timer.start(500)  # 再等500ms
                return

            # 检查相机线程状态
            if hasattr(self.new_thread, 'use_orbbec'):
                if self.new_thread.use_orbbec:
                    self.statusbar.showMessage("✅ Orbbec 3D相机已连接，实时预览已启动")
                    camera_info = """
🎥 相机信息:
• 设备类型: Orbbec 3D相机
• 分辨率: 640x480
• 深度检测: 支持
• 状态: 正常运行

📊 系统状态:
• GPU加速: """ + ("启用" if GPU_DEVICE else "禁用") + """
• AI模型: 已加载
• 实时预览: 运行中
                    """
                else:
                    self.statusbar.showMessage("✅ USB摄像头已连接，实时预览已启动")
                    camera_info = """
🎥 相机信息:
• 设备类型: USB摄像头
• 分辨率: 640x480
• 帧率: 30fps
• 状态: 正常运行

📊 系统状态:
• GPU加速: """ + ("启用" if GPU_DEVICE else "禁用") + """
• AI模型: 已加载
• 实时预览: 运行中
                    """

                # 显示相机信息
                self.ResultLabel.setText(camera_info)
                self.label.setText("📹 相机预览运行中 - 点击'开始检测'进行目标识别")

                # 启用开始按钮
                self.StartButton.setVisible(True)
                self.StartButton.setEnabled(True)

            else:
                # 相机初始化失败的情况
                self.statusbar.showMessage("❌ 相机连接失败")
                self.label.setText("❌ 相机连接失败")
                error_info = """
❌ 相机连接失败

🔧 请检查:
• USB连接是否正常
• 相机驱动是否安装
• 设备是否被其他程序占用
• Orbbec SDK是否正确安装

💡 解决方案:
1. 重新插拔USB连接
2. 关闭其他使用相机的程序
3. 重启应用程序
4. 检查设备管理器中的相机设备
                """
                self.ResultLabel.setText(error_info)

        except Exception as e:
            self.statusbar.showMessage(f"❌ 相机测试失败: {str(e)}")
            self.label.setText("❌ 相机测试失败")
            print(f"相机自动测试失败: {e}")



    def OpenImage(self):
        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap('inference\\images\\WIN_20201004_19_48_41_Pro.jpg')
        # self.label.setPixmap(jpg)
        self.ImgLabel.setPixmap(jpg)
        self.ResultLabel.setText('hello world!')
        self.label.setText("🔍 正在检测中...")

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
        self.thread_run = False
        # self.fri_pipeline.stop()
        # 流式传输循环

        # 记录帧数

    def StartButton_clicked(self):
        self.detect()


if __name__ == '__main__':  # 程序的入口

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
