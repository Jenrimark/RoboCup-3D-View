import math
import sys
import pyransac3d as pyrsc
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtWidgets import QMainWindow
from td_recognition import Ui_MainWindow  # 加载我们的布局
from threading import Thread
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


import pyrealsense2 as rs
import pyorbbecsdk as orsdk
import numpy as np
import cv2
import copy
import argparse
import os
# import platform
import shutil
import time
# from pathlib import Path
# import math
import random

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
#from sklearn.linear_model import LinearRegression
import socket

import argparse

from models.common import DetectMultiBackend
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams

from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box

#最后总的物品的个数
last_number=[]
for i in range(20):
    last_number.append(0)
#识别时间
detect_time=[16,16,16]

# #待转向中间的间隔时间
sleep_time=9

paper = 0.01

ground_dis = 0.2

SAT_NUM=0.6
#误判斜率
xielv=1
#轮数 这里只能取 2，4


# 删除平面中离群点的阈值
num_p = 120

# 删除拟合平面的阈值
num_m = 101
address = '192.168.174.36'
# 统计个数
#单轮的物品个数
number = []
one_round = []
for i in range(20):
    number.append(0)
    one_round.append(0)
myList = [([0] * 100) for i in range(20)]

radio = [[0.75, 0.25, 0.5, 0.5],
         [0.5, 0.5, 0.25, 0.75],
         [0.25, 0.75, 0.5, 0.5],
         [0.5, 0.5, 0.75, 0.25],
         [0.5, 0.5, 0.5, 0.5]]

data ={0:'CA001', 1:'CA002', 2:'CA003', 3:'CA004', 4:'CB001', 5:'CB002', 6:'CB003', 7:'CB004',
       8:'CC001', 9:'CC002', 10:'CC003', 11:'CC004',12:'CD001',13:'CD002', 14:'CD003', 15:'CD004',
       16:'W001',17:'W002',18:'W003',19:'W004'}
client = socket.socket()
class Protocol:

    def __init__(self, bs=None):
        self.bs = bytearray(0)

    def add_int32(self, val):#添加dataType（0：开始计时 1：发送识别结果 2：发送测量结果）
        bytes_val = bytearray(val.to_bytes(4, byteorder='big'))
        self.bs += bytes_val

    def add_str(self, val):#添加需要发送的字符串data(字符长度计算已包含在内)
        bytes_val = bytearray(val.encode(encoding='utf8'))
        bytes_length = bytearray(len(bytes_val).to_bytes(4, byteorder='big'))
        self.bs += (bytes_length + bytes_val)
    def pack_send(self,dtype,con):
        self.add_int32(dtype)
        self.add_str(con)

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


class new_thread_(QThread):

    def __init__(self, Window):
        super(new_thread_, self).__init__()
        self.window = Window

    def run(self):
        time.sleep(sleep_time)
        self.window.thread_run=False

class detect_Flag_thread(QThread):

    def __init__(self,window):
        super(detect_Flag_thread, self).__init__()
        self.window=window

    def run(self):
        time.sleep(detect_time[self.window.round])
        self.window.detect_Flag=False

    
class Predictor:
    def __init__(self,model_path,device,conf_thres=0.25,iou_thres=0.45,max_det=1000):    
        # Load model
        self.segment_model = DetectMultiBackend(model_path, device=device, dnn=False, data='', fp16=False)
        self.conf_thres = conf_thres
        self.iou_thres=iou_thres
        self.max_det = max_det
    def predict(self,im0,times):
        imgsz=(640, 480);
        #im0 = cv2.resize(im0,(320,240))#,interpolation=cv2.INTER_LINEAR
        stride, names, pt = self.segment_model.stride, self.segment_model.names, self.segment_model.pt
        dataset = LoadImages(im0, img_size=imgsz, stride=stride, auto=pt, vid_stride=1);dt=(Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.segment_model.device)
                im = im.half() if self.segment_model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255 
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
    
            # Inference
            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                
                pred, proto = self.segment_model(im, augment=False, visualize=False)[:2]
    
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,None, False, max_det=self.max_det, nm=32);break
    
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
    
            # Process predictions
        findet=[]
        max_conf = -1
        for i, det in enumerate(pred):  # per image 
            for j,(*x,conf,y) in enumerate(reversed(det[:,:6])):
                if conf>max_conf:
                    findet = det[i]
        if(type(findet)==list):
            masks = [[[0 for _ in range(640)]for _ in range(480)]]
            masks = np.array(masks)
            masks[180:300][240:400] = 1 
            return masks;
        det = findet.unsqueeze(0)
        
        annotator = Annotator(im0, line_width=3, example="exp")
                
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
        #masks[0] = cv2.resize(masks[0],imgsz)#,interpolation=cv2.INTER_LINEAR
        

        '''with open('a.txt', 'w') as file:
            for row in masks[0]:
                row_str = ' '.join([str(value.item()) for value in row])  # 将张量的行转换为字符串
                file.write(row_str + '\n')  # 写入每一行数据到文件'''


        #print(masks[masks[:, :, :] > 0])
        if times==0:
            annotator.masks(
                masks,
                colors=[colors(x, True) for x in det[:, 5]],
                im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.segment_model.device).permute(2, 0, 1).flip(0).contiguous() /
                255 if False else im[i])
            im0 = annotator.result()
            cv2.imwrite("cjq.jpg", im0)
        #cv2.imshow("cjq", im0)
        #cv2.waitKey(0)
        #cv2.destoryAllWindows()
        
        return masks

def check_in_desk(desk,p):
    if p[1]>desk[2][1] or p[1]<desk[1][1]:
        return False
    #print("check 1")
    if abs(p[0] - desk[2][0]) < 0.001:
        k1 = 1
    else:
        k1 = (p[1]-desk[2][1])/(p[0]-desk[2][0])
    k_desk_l = (desk[0][1]-desk[2][1])/(desk[0][0]-desk[2][0])
    if k1>0 or k1<k_desk_l:
        return False
    #print("check 2")
    if abs(p[0] - desk[3][0]) < 0.001:
        k2=-1
    else:
        k2 = (p[1]-desk[3][1])/(p[0]-desk[3][0])
    k_desk_r = (desk[1][1]-desk[3][1])/(desk[1][0]-desk[3][0])
    if k2<0 or k2>k_desk_r:
        return False
    #print("check 3")
    return True

class new_thread(QThread):

    def __init__(self, Window):
        super(new_thread, self).__init__()
        self.window = Window
        self.pipeline = orsdk.Pipeline()
        self.round=0
        self.f = True
      
        # 平面方程参数
        state = AppState()
        # Create a config and configure the pipeline to stream
        config = orsdk.Config()
       # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度图像，分辨率(640, 480)，帧率30
       # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB彩色图像，分辨率(640, 480)，帧率30
        color_profile_list = self.pipeline.get_stream_profile_list(orsdk.OBSensorType.COLOR_SENSOR) 
        color_profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        depth_profile_list = self.pipeline.get_stream_profile_list(orsdk.OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        cudnn.benchmark = True

        # 开始流式传输
        profile = self.pipeline.start(config)
        # 修改相机内参
        '''
        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]
        sensor.set_option(rs.option.motion_range, 62)
        sensor.set_option(rs.option.filter_option, 3)
        sensor.set_option(rs.option.confidence_threshold, 9)
        # 默认参数
        print(rs.options.get_option_range(sensor, rs.option.visual_preset))
        print(rs.options.get_supported_options(sensor))
        #sensor.set_option(rs.option.visual_preset, 12)
        sensor.set_option(rs.option.laser_power, 13)
        #sensor.set_option(rs.option.accuracy, 14)
        sensor.set_option(rs.option.frames_queue_size, 19)
        #sensor.set_option(rs.option.depth_units, 28)

        # profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        # Processing blocks
        self.pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

        self.colorizer = rs.colorizer()

        # 获取深度传感器的深度标尺
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        # 删除对象的背景
        clipping_distance_in_meters = 1.6  # 1 meter
        self.clipping_distance = clipping_distance_in_meters / depth_scale
        print("Clipping Distance is: ", self.clipping_distance)

        # 创建对齐对象
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        '''
        config.set_align_mode(orsdk.OBAlignMode.HW_MODE)



        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # 初始化相关参数
        set_logging()
        self.device = select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # 加载yolo模型
        self.model = attempt_load(weights, device=self.device)
        if self.half:
            self.model.half()  # to FP16

        # 加载nanodet模型
        #torch.backends.cudnn.enabled = False
        #torch.backends.cudnn.benchmark = False
        '''load_config(cfg, "NanoDet/config/nanodet-m.yml")
        logger_nano = Logger(-1, use_tensorboard=False)
        self.predictor_nanodet = Predictor(cfg, "NanoDet/tools/workspace/nanodet_m/model_best/model_best.pth", logger_nano, device=self.device)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]'''
         # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.predictor=Predictor('best-seg.pt',self.device)

        self.model_soap = attempt_load('best-detect.pt',device=self.device)
        if self.half:
            self.model_soap.half()  # to FP16      


    def run(self):

        times = 0
        seconds_per = 3
        start = time.time()
        print('show camera scene....')
        #client.connect((address, 6666))
        #color_image2 = cv2.imread("333.jpg")
        while True:
            frames = self.pipeline.wait_for_frames(500)
            camera_param = self.pipeline.get_camera_param();fx = camera_param.rgb_intrinsic.fx;fy = camera_param.rgb_intrinsic.fy;cx = camera_param.rgb_intrinsic.cx;cy = camera_param.rgb_intrinsic.cy
            # 深度图像对齐到彩色图像
            #aligned_frames = self.align.process(frames)

            # 获取对齐前的深度图像和彩色图像的帧
            #depth_frame = frames.get_depth_frame()
            #color_frame = frames.get_color_frame()
            # 获取对齐后的深度图像和彩色图像的帧
            
            
            #aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            #aligned_color_frame = aligned_frames.get_color_frame()
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            #depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            # 验证是否真的获取到了图像
            #if not depth_frame or not color_frame or not aligned_depth_frame or not aligned_color_frame:
            #    continue
            #if not aligned_depth_frame or not aligned_color_frame:
            #    continue
            if not depth_frame or not color_frame:
                continue
            # 将图像转换为numpy数组类型
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = self.color_frame_to_bgr_img0(color_frame)
            
            
            
            #color_image = copy.deepcopy(color_image2)
            #color_image = cv2.resize(color_image,(640,480))
            
            
            
            img_cpy = color_image.copy()

            img0 = color_image.copy()

            # 将图像转化为网络输入需要的形状
            # Letterbox变换，这里网络输入需要（640，512）,而图像为（640，480）,因此需要转换。
            img = self.window.letterbox(img0, new_shape=640, auto=True)[0]
            # 维度变换
            img = np.expand_dims(img, 0)
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
            img = np.ascontiguousarray(img)

            # 将图像转换为张量
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            if self.window.thread_run:

                # depth_image = np.asanyarray(aligned_depth_frame.get_data())
                masks=self.predictor.predict(img0,1)

                img_display = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)
                img_display = QImage(img_display, img_display.shape[1], img_display.shape[0], img_display.shape[1] * 3,
                                     QImage.Format_RGB888)

                img_display = QtGui.QPixmap(img_display).scaled(640, 480)
                self.window.ImgLabel.setPixmap(img_display)
                self.window.label.setText(self.window._translate("MainWindow",
                                                                 "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">空闲</span></p></body></html>"))
                pred = self.model(img, augment=opt.augment)[0]
                pred = self.model_soap(img, augment=opt.augment)[0]
                self.window.StartButton.setVisible(True)
                
            else:
                with torch.no_grad():
                # 预测，返回的结果包括识别的物体类别，框的位置，和该类物品的概率
                    if times%seconds_per!=0:
                        times+=1
                        continue
                    pred = self.model(img, augment=opt.augment)[0]
                    pred_soap = self.model_soap(img, augment=opt.augment)[0]
                    # 使用NMS对前面网络预测的结果进行过滤
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                    pred_soap = non_max_suppression(pred_soap, 0.8, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                    det = pred[0]
                    det_soap = pred_soap[0]
                    if det is not None and len(det):
                        # 将网络输入的时候变换成的(640,512)的图像中的框的坐标转换为(640，480)中的框的坐标。
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], color_image.shape).round()
                        det = det.cpu()
                        det = det.numpy()
                        det = torch.from_numpy(det).cuda()  # 平台移植的时候一定要修改
                    if det_soap is not None and len(det_soap):
                        # 将网络输入的时候变换成的(640,512)的图像中的框的坐标转换为(640，480)中的框的坐标。
                        det_soap[:, :4] = scale_coords(img.shape[2:], det_soap[:, :4], color_image.shape).round()
                        det_soap = det_soap.cpu()
                        det_soap = det_soap.numpy()
                        det_soap = torch.from_numpy(det_soap).cuda()  # 平台移植的时候一定要修改
                if times == 0:
                    start = time.time()
                    self.detect_Flag = True
                    self.detect_new_thread = detect_Flag_thread(self)
                    self.detect_new_thread.start()
                    p_start = Protocol()
                    p_start.pack_send(0, 'cugrobot')
                    print("send",time.time()-start)
                    #client.send(p_start.bs)
                    # mapped_frame, color_source = color_frame, color_image
                    # self.pc.map_to(aligned_color_frame)
                    # points = self.pc.calculate(aligned_depth_frame)
                    # v = points.get_vertices() # pyrealsense2.pyrealsense2.BufData
                    # verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
                    # # np.savetxt("scene.xyz", verts)
                    # pt = []
                    # for i in range(0, len(verts)):
                    #     if verts[i][2] < 0.0001 or verts[i][2] > 1.2:
                    #         continue
                    #     pt.append([verts[i][0], -verts[i][1], -verts[i][2]])
                    # pt = np.array(pt)
                    # plane1 = pyrsc.Plane()
                    # plane_model, inliers = plane1.fit(pt, 0.01, maxIteration=500)
                    # [self.plane_a, self.plane_b, self.plane_c, self.plane_d] = plane_model
                    # print(f"Plane equation: {self.plane_a:.2f}x + {self.plane_b:.2f}y + {self.plane_c:.2f}z + {self.plane_d:.2f} = 0")
                    #self.pc.map_to(aligned_color_frame)
                    #points = self.pc.calculate(aligned_depth_frame)
                    #v = points.get_vertices()  # pyrealsense2.pyrealsense2.BufData
                    
                    #verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
                    #points = frames.convert_to_points(camera_param)
                    #np.savetxt("scene.xyz", verts)
                    '''pt = []
                    for i in range(len(points)):
                        pt.append([points[i].x, -points[i].y, -points[i].z])
                    pt = np.array(pt)
                    plane1 = pyrsc.Plane()
                    plane_model, inliers = plane1.fit(pt, 0.01, maxIteration=1000)
                    [self.plane_a, self.plane_b, self.plane_c, self.plane_d] = plane_model
                    print(
                        f"Plane equation: {self.plane_a:.2f}x + {self.plane_b:.2f}y + {self.plane_c:.2f}z + {self.plane_d:.2f} = 0")'''
            ############################ nanodet 获得桌子位置 ############################
                    #masks = self.predictor.predict(img0);print('predict finish!!!')
                    #print("mask",time.time()-start)
                    
                    
                    
                    '''img0=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
                    img0=np.repeat(img0[:,:,np.newaxis],3,axis=2)
                #print(img0.shape)
                #print(img0)
                    meta, res = self.predictor_nanodet.inference(img0)
                #print(res)
                    score_thresh = 0.3
                    all_box = []
                    for label in res:
                        for bbox in res[label]:
                            score = bbox[-1]
                            if score > score_thresh:
                                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                                all_box.append([x0, y0, x1, y1, score])
                    all_box.sort(key=lambda v: v[4])
                    desk = None
                    if len(all_box) > 0:
                        desk = all_box[0]
                    ###########################'''

                masks = self.predictor.predict(img0,times);print('predict finish!!!')
                
                    
                '''if masks == None:
                    times += 1
                    print('no masks,continue')
                    continue;'''
                print("mask",time.time()-start)
                if det is None or len(det) == 0:
                    times += 1
                    continue
                    
                one_round.clear()
                for i in range(20):
                    one_round.append(0)

                # 过滤地面的物品和打印物品
                # xy_center = []
                for counter in range(det.shape[0]):
                    x1, y1, x2, y2, _, cls = det[counter]
                    if int(cls) ==2:
                        continue
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    # 对每一帧传出的画框结果进行处理 x1,y1,x2,y2
                    #color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

                    x_center = int((x1 + x2) / 2)
                    y_center = int(0.75*y2+0.25*y1)

                    ###################检测物品的中心是否在桌子里面###################
                    desk_cnt = 0
                    if len(masks)>0:
                        # if (desk[0] < x_center < desk[2] and desk[1] < y_center < desk[3]):
                        if masks[0][y_center][x_center]>0:
                            # desk_cnt = desk_cnt + 1
                            item = int(det[counter, 5])
                            one_round[item] += 1
                        else:
                            det[counter][-1]=-1
                            
                for *xyxy, conf, cls in (det):
                    if int(cls) == -1:
                        continue
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, color_image, label=label, color=self.colors[int(cls)], line_thickness=1)
                    
                #####soap#####class = 0
                if det_soap is not None and len(det_soap) != 0:
                    for counter in range(det_soap.shape[0]):
                        x1, y1, x2, y2, _, _ = det_soap[counter]
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        # 对每一帧传出的画框结果进行处理 x1,y1,x2,y2
                        #color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
    
                        x_center = int((x1 + x2) / 2)
                        y_center = int(0.75*y2+0.25*y1)
    
                        ###################检测物品的中心是否在桌子里面###################
                        if len(masks)>0:
                            # if (desk[0] < x_center < desk[2] and desk[1] < y_center < desk[3]):
                            if masks[0][y_center][x_center]>0:
                                # desk_cnt = desk_cnt + 1
                                item = 2
                                one_round[item] += 1
                            else:
                                det_soap[counter][-1]=-1
                                
                    for *xyxy, conf, cls in (det_soap):
                        if int(cls) == -1:
                            continue
                        cls = '2'
                        label = '%s %.2f  small' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, color_image, label=label, color=self.colors[int(cls)], line_thickness=1)

                print('panduan',time.time()-start)
                # 在平面上画出物品
                img_display = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                img_display = QImage(img_display, img_display.shape[1], img_display.shape[0], img_display.shape[1] * 3,
                                     QImage.Format_RGB888)
                img_display = QtGui.QPixmap(img_display).scaled(640, 480)
                self.window.ImgLabel.setPixmap(img_display)

                self.window.label.setText(self.window._translate("MainWindow",
                                                                 "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">检测中</span></p></body></html>"))

                # self.window.label.repaint()
                '''for k in range(16):
                    if one_round[k]>myList[k][0]:
                        myList[k][0]=one_round[k]
                        number[k]=myList[k][0]'''
                print(one_round)
                for k in range(20):
                    myList[k][one_round[k]] = myList[k][one_round[k]] + 1
                
                if times==150:
                    for i in range(20):
                        max_ = 0
                        k = 0
                        if myList[i][0] < times/seconds_per * SAT_NUM:
                            k = 1
                        for j in range(k, 100):
                            if max_ < myList[i][j]:
                                max_ = myList[i][j]
                                number[i] = j
                
                if times>=150:
                    print("finish!!!!!!",time.time()-start)
                    break
                times += 1

        self.window.label.setText(self.window._translate("MainWindow",
                                                         "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">检测完成</span></p></body></html>"))
        # self.window.label.repaint()
        # print('pause', myList)
        pri = []
        pri2 = []
        # print('number',type(number[19]))
        result_str = ''
        result_str2 = ''
        for aa in range(20):
            
            if number[aa] != 0:
                # print("number[aa]",number[aa])
                st = "目标ID：" + str(data[aa]) + "   数量：" + str(number[aa])
                # print("st",st)
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
        #client.send(p_end.bs)

        filename = "/home/cug/Desktop/reco_result/CUG-CUG_Robot-R1" + ".txt"
        f = open(filename, 'w+')
        # os.linesep代表当前操作系统上的换行符
        f.write('START' + '\n')
        f.write(result_str2)
        f.write('END' + os.linesep + '\n')
        f.close()

        self.window.ResultLabel.setText(result_str)
        print('finish',time.time()-start)
        # cli
        # self.window.ResultLabel.repaint()
        
    def color_frame_to_bgr_img0(self, frame):
        '''将彩图数据帧转换为numpy格式的BGR彩图'''
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        image = np.zeros((height, width, 3), dtype=np.uint8)
        if color_format == orsdk.OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == orsdk.OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            print("不支持彩图数据格式: {}".format(color_format))
            return None
        return image


class UsingTest(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(UsingTest, self).__init__(*args, **kwargs)
        self.setupUi(self)  # 初始化u
        self._translate = QtCore.QCoreApplication.translate
        self.setWindowTitle('3D识别 第 1'+'轮')
        self.setWindowIcon(QIcon("icon/cug.ico"))
        self.thread_run = True
        self.new_thread = new_thread(self)
        self.new_thread.start()
        self.StartButton.clicked.connect(self.detect)

        # self.wait_start = Receive(self)
        # self.wait_start.start()  # open the receive signal thread
        #jpg = QtGui.QPixmap('inference\\images\\start.jpg')
        # self.ImgLabel.setPixmap(jpg)
        #self.model_inti()

        #self.end_=False
        #self.per_view()
        #t=Thread(target=self.view())
        #t.start()
        #self.view()



    def OpenImage(self):
        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap('inference\\images\\WIN_20201004_19_48_41_Pro.jpg')
        # self.label.setPixmap(jpg)
        self.ImgLabel.setPixmap(jpg)
        self.ResultLabel.setText('hello world!')
        self.label.setText(self._translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">检测中</span></p></body></html>"))


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
        self.thread_run=False
        #self.fri_pipeline.stop()
        # 流式传输循环

            # 记录帧数


    def StartButton_clicked(self):
        self.detect()
	








if __name__ == '__main__':  # 程序的入口

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best-detect.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
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
