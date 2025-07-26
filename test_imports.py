#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试各个模块的导入情况
用于定位DLL加载失败的具体原因
"""

import sys
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")
print("-" * 50)

# 测试基础模块
modules_to_test = [
    ("sys", "系统模块"),
    ("os", "操作系统模块"),
    ("random", "随机数模块"),
    ("time", "时间模块"),
    ("argparse", "参数解析模块"),
    ("socket", "网络模块"),
    ("shutil", "文件操作模块"),
]

print("🔍 测试基础Python模块:")
for module_name, description in modules_to_test:
    try:
        __import__(module_name)
        print(f"✅ {module_name} ({description}) - 导入成功")
    except Exception as e:
        print(f"❌ {module_name} ({description}) - 导入失败: {e}")

print("\n" + "-" * 50)

# 测试科学计算模块
scientific_modules = [
    ("numpy", "NumPy数值计算"),
    ("cv2", "OpenCV图像处理"),
    ("torch", "PyTorch深度学习"),
    ("torchvision", "PyTorch视觉"),
]

print("🔬 测试科学计算模块:")
for module_name, description in scientific_modules:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {module_name} ({description}) - 版本: {version}")
    except Exception as e:
        print(f"❌ {module_name} ({description}) - 导入失败: {e}")

print("\n" + "-" * 50)

# 测试PyQt5模块
pyqt_modules = [
    ("PyQt5", "PyQt5基础"),
    ("PyQt5.QtCore", "PyQt5核心"),
    ("PyQt5.QtGui", "PyQt5图形界面"),
    ("PyQt5.QtWidgets", "PyQt5控件"),
]

print("🖥️ 测试PyQt5模块:")
for module_name, description in pyqt_modules:
    try:
        __import__(module_name)
        print(f"✅ {module_name} ({description}) - 导入成功")
    except Exception as e:
        print(f"❌ {module_name} ({description}) - 导入失败: {e}")

print("\n" + "-" * 50)

# 测试专业模块
specialized_modules = [
    ("ultralytics", "YOLO模型"),
    ("pyorbbecsdk", "Orbbec相机SDK"),
]

print("🎯 测试专业模块:")
for module_name, description in specialized_modules:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {module_name} ({description}) - 版本: {version}")
    except Exception as e:
        print(f"❌ {module_name} ({description}) - 导入失败: {e}")

print("\n" + "-" * 50)

# 测试CUDA支持
print("🚀 测试CUDA支持:")
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    else:
        print("使用CPU模式")
except Exception as e:
    print(f"❌ PyTorch CUDA测试失败: {e}")

print("\n" + "-" * 50)

# 测试OpenCV功能
print("📷 测试OpenCV功能:")
try:
    import cv2
    print(f"OpenCV版本: {cv2.__version__}")
    
    # 测试创建图像
    import numpy as np
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    print("✅ 图像创建测试通过")
    
    # 测试颜色转换
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    print("✅ 颜色转换测试通过")
    
except Exception as e:
    print(f"❌ OpenCV功能测试失败: {e}")

print("\n" + "=" * 50)
print("🎉 模块导入测试完成!")
print("请查看上面的结果，找出导入失败的模块")
