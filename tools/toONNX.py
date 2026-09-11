# 参考：将 PyTorch 模型导出为 ONNX 格式
# from ultralytics import YOLO
#
# # 加载YOLO模型
# model = YOLO("weights/best.pt")
#
# # 将模型导出为ONNX格式
# success = model.export(format="onnx", simplify=True)
#
# # 检查是否成功导出
# assert success


from ultralytics import YOLO
from PIL import Image

# 使用当前正式模型做推理演示（导出 ONNX 后也可换回 .onnx 文件）
model = YOLO("weights/best.pt")
results = model(source=r"assets/P_image_51.jpg")

# 展示结果
for result in results:
    boxes = result.boxes  # 边界框输出的 Boxes 对象
    print("boxes： ", boxes)
    # 可以使用Result对象的plot()方法来可视化预测结果。它会将Results对象中包含的所有预测类型（框、掩码、关键点、概率等）绘制到一个numpy数组上，然后可以显示或保存。
    im_array = result.plot()  # 绘制包含预测结果的BGR numpy数组
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
    im.show()  # 显示图像
    im.save('result_output/results.jpg')  # 保存图像
pass
