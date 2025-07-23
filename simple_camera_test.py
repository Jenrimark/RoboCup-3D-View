#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的相机切换逻辑测试
不需要实际的相机硬件，只测试导入和逻辑
"""

def test_camera_switching():
    """测试相机切换逻辑"""
    print("=== 相机切换逻辑测试 ===")
    
    # 测试Orbbec SDK导入
    try:
        import pyorbbecsdk as orsdk
        print("✓ pyorbbecsdk 导入成功")
        orbbec_available = True
    except ImportError as e:
        print(f"✗ pyorbbecsdk 导入失败: {e}")
        print("  这是正常的，如果没有安装Orbbec SDK")
        orbbec_available = False
    
    # 测试OpenCV导入
    try:
        import cv2
        print("✓ OpenCV 导入成功")
        opencv_available = True
    except ImportError as e:
        print(f"✗ OpenCV 导入失败: {e}")
        opencv_available = False
    
    # 模拟相机初始化逻辑
    print("\n=== 模拟相机初始化 ===")
    
    use_orbbec = False
    
    if orbbec_available:
        print("尝试初始化Orbbec相机...")
        # 模拟Orbbec相机初始化失败（因为没有硬件）
        try:
            # 这里会失败，因为没有实际的Orbbec相机硬件
            pipeline = orsdk.Pipeline()
            print("✓ Orbbec相机初始化成功")
            use_orbbec = True
        except Exception as e:
            print(f"✗ Orbbec相机初始化失败: {e}")
            print("  这是正常的，因为没有连接Orbbec相机硬件")
    
    if not use_orbbec and opencv_available:
        print("切换到普通摄像头...")
        try:
            # 模拟普通摄像头初始化
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✓ 普通摄像头初始化成功")
                cap.release()
            else:
                print("✗ 普通摄像头初始化失败")
        except Exception as e:
            print(f"✗ 普通摄像头初始化失败: {e}")
    
    # 总结
    print("\n=== 测试总结 ===")
    if orbbec_available:
        print("✓ 系统支持Orbbec相机（SDK已安装）")
    else:
        print("✗ 系统不支持Orbbec相机（SDK未安装）")
    
    if opencv_available:
        print("✓ 系统支持普通摄像头（OpenCV已安装）")
    else:
        print("✗ 系统不支持普通摄像头（OpenCV未安装）")
    
    print(f"\n推荐的相机类型: {'Orbbec相机（如果硬件可用）' if orbbec_available else '普通摄像头'}")
    
    return orbbec_available, opencv_available

def show_installation_guide():
    """显示安装指南"""
    print("\n=== 安装指南 ===")
    print("如果需要安装缺失的依赖，请运行以下命令：")
    print()
    print("安装OpenCV:")
    print("  pip install opencv-python")
    print()
    print("安装PyQt5 (用于GUI):")
    print("  pip install PyQt5")
    print()
    print("安装其他依赖:")
    print("  pip install torch torchvision ultralytics numpy")
    print()
    print("安装Orbbec SDK:")
    print("  请从Orbbec官网下载并安装pyorbbecsdk")

if __name__ == "__main__":
    orbbec_ok, opencv_ok = test_camera_switching()
    
    if not opencv_ok:
        show_installation_guide()
    
    print("\n测试完成！")
