# RoboCup-3D

RoboCup 3D 仿真足球机器人视觉检测系统 —— 基于 Orbbec 深度相机 + YOLO（ultralytics）+ NanoDet 的实时目标检测与识别程序，面向 RoboCup 3D 仿真组比赛。

本项目为比赛实际使用的视觉检测程序：通过 Orbbec 深度相机采集赛场画面，使用 YOLO 模型完成目标检测 / 分割 / 颜色识别，PyQt5 图形界面实时显示检测结果，并通过 Socket 与机器人决策程序通信，将识别结果发送给场上机器人。

> 说明：本项目由团队历届比赛代码演进而来，仓库中保留了完整的版本迭代历史（见 `archive/` 与 Git 提交记录）。当前维护入口为 `pydect3_2025.py` / `pydect4_2025.py`。

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [安装部署](#安装部署)
- [模型权重](#模型权重)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
- [版本演进](#版本演进)
- [开发者指南](#开发者指南)
- [常见问题](#常见问题-faq)
- [致谢](#致谢)

---

## 功能特性

- **实时目标检测**：基于 ultralytics YOLO，支持检测 / 实例分割 / 颜色分类三种任务
- **多模型协同**：主检测模型 + 分割模型 + 颜色识别模型 + 水果模型（历史版本），按需加载
- **Orbbec 深度相机支持**：优先使用 Orbbec 相机，失败自动回退普通摄像头
- **图形化界面**：PyQt5 深色主题界面，实时显示摄像头画面、检测框、状态信息
- **Socket 通信**：检测结果通过 TCP 发送给机器人端（默认端口 6666）
- **轻量级备选方案**：内置 NanoDet 子项目，用于嵌入式 / 无 GPU 场景的快速检测

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RoboCup-3D 视觉检测系统                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Orbbec 深度相机 ──► 视频帧采集（CameraThread）                  │
│                              │                                   │
│                              ▼                                   │
│                   YOLO 检测/分割/颜色识别                         │
│                   （主检测 best.pt + 分割 yuan0517.pt 等）         │
│                              │                                   │
│                              ▼                                   │
│                    PyQt5 图形界面（td_recognition.py）            │
│                    ├─ 实时画面 + 检测框绘制                       │
│                    └─ 状态栏 / 结果文本 / 重新检测                 │
│                              │                                   │
│                              ▼                                   │
│               Socket TCP 通信（NetworkThread, 端口 6666）         │
│                              │                                   │
│                              ▼                                   │
│                    机器人决策 / 上位机（外部程序）                 │
└─────────────────────────────────────────────────────────────────┘

硬件依赖：
- Orbbec 深度相机（如 Gemini 系列；SDK 见 3rdparty/orbbec_sdk/）
- 或普通 USB 摄像头（自动回退）
- 建议 x86_64 Windows / Linux；无 GPU 可运行（默认 CPU 模式）
```

**软件流程**（主程序 `pydect3_2025.py`）：

1. 启动时加载主检测模型 `weights/best.pt` 与分割模型 `weights/yuan0517.pt`（后台线程加载，避免界面卡顿）
2. `CameraThread` 采集视频帧（Orbbec → 普通摄像头自动切换）
3. 每帧执行目标检测 / 分割，绘制检测框与类别标签
4. 结果按既定协议打包，通过 `NetworkThread` 发送至机器人（默认 `IP: 6666`）
5. 界面支持手动"重新检测"、显示连接状态与检测结果

> 检测类别与发送协议见代码头部 `data` 字典及 `pack_send()` 实现。

---

## 目录结构

整理后的仓库布局（2025-09 规范化）：

```
RoboCup-3D/
├── README.md                     # 本文档
├── .gitignore                    # 忽略权重/二进制/缓存/日志
├── requirements.txt              # Python 依赖（Python 3.9）
│
├── pydect3_2025.py               # 【当前主程序 v3】检测 + 分割
├── pydect4_2025.py               # 【当前主程序 v4】检测 + 分割 + 颜色识别
├── td_recognition.py             # PyQt5 图形界面定义（美化版，被主程序 import）
├── start_with_conda.bat          # Windows 一键启动脚本（检查环境与权重）
│
├── camera_test.py                # 相机切换测试（Orbbec ↔ 普通摄像头）
├── simple_main.py                # 简化版主程序（仅界面 + 摄像头）
├── simple_camera_test.py         # 相机切换逻辑测试（无需硬件）
├── test_imports.py               # 依赖导入诊断工具
├── toONNX.py                     # YOLO ONNX 模型推理演示
│
├── weights/                      # ★ 全部模型权重（.pt / .onnx）集中管理
│   ├── README.md                 # 权重清单与用途说明（重要，见下）
│   ├── best.pt                   # 主检测模型（当前版本使用）
│   ├── yuan0517.pt               # 分割模型（当前版本使用）
│   └── ...                       # 其余训练产物，详见 weights/README.md
│
├── 3rdparty/
│   └── orbbec_sdk/               # Orbbec 相机 SDK 动态库（不入库）
│       ├── win/                  #   Windows: OrbbecSDK.dll / .lib / .pyd ...
│       └── linux/                #   Linux: libOrbbecSDK.so* / libdepthengine.so* ...
│
├── models/                       # ultralytics YOLOv5 模型定义包（含 hub/、segment/）
├── utils/                        # ultralytics 工具包（dataloaders、metrics、plots 等）
├── NanoDet/                      # NanoDet-PyTorch 轻量检测子项目（独立、自含权重）
├── archive/                      # 历史版本脚本存档（2023/2024/2025 快照）
│   └── README.md                 # 各版本说明
│
├── assets/                       # 测试图片与示例输出（cjq.jpg、P_image_51.jpg 等）
├── docs/
│   ├── 界面美化详细说明.md         # UI 美化改造说明
│   └── environment/              # 环境记录（requirements_remove、pip 日志等）
├── icon/                         # 程序图标（cug.ico、TurnImg.png）
└── result_output/                # 程序输出目录（含连接结果示例）
```

---

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11（主）、Linux（测试通过） |
| Python | 3.8 ~ 3.9（`requirements.txt` 基于 3.9 锁定） |
| 相机 | Orbbec 深度相机（可选，无则回退普通摄像头） |
| GPU | 可选，默认 CPU 模式运行（`GPU_DEVICE = False`） |
| 依赖体积 | PyTorch 2.2.0 + CUDA（可选）/ CPU |

---

## 安装部署

### 1. 创建 Conda 环境（推荐）

```bash
conda create -n robocup3d python=3.9 -y
conda activate robocup3d
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 需要 GPU 加速时，将 PyTorch 替换为对应 CUDA 版本：
> ```bash
> pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. 安装 Orbbec SDK（使用深度相机时）

- 方式一：`pip install pyorbbecsdk`（推荐，自动携带动态库）
- 方式二：将 `3rdparty/orbbec_sdk/<平台>/` 下的动态库放置到程序运行目录或系统库路径
  - Windows：`OrbbecSDK.dll`、`depthengine_2_0.dll` 需与 `pyorbbecsdk.cp39-win_amd64.pyd` 同目录或加入 `PATH`
  - Linux：将 `libOrbbecSDK.so*`、`libdepthengine.so*` 加入 `LD_LIBRARY_PATH`

### 4. 准备权重

模型权重因体积较大（合计约 523 MB）**不入库管理**，请从以下渠道获取并放入 `weights/` 目录：

- 团队共享网盘 / 本地备份（推荐，含全部历史权重）
- GitHub Releases 附件（如已发布）
- 使用自有数据集重新训练（见[开发者指南](#训练新模型)）

> 当前主程序最少需要的权重：`weights/best.pt`（检测）与 `weights/yuan0517.pt`（分割）。

---

## 模型权重

所有权重集中存放于 [`weights/`](./weights/) 目录，**请先阅读 [`weights/README.md`](./weights/README.md)**，其中包含：

- 每个权重文件的用途（主检测 / 分割 / 颜色分类 / 水果检测 / 实验版本）
- 对应使用它的脚本（当前版 / 历史版）
- 文件大小与获取方式

---

## 快速开始

### 运行主程序（推荐）

```bash
# 从仓库根目录运行（权重路径为相对路径 weights/xxx.pt）
python pydect3_2025.py        # v3：检测 + 分割
python pydect4_2025.py        # v4：检测 + 分割 + 颜色识别
```

Windows 下可直接双击 `start_with_conda.bat`（自动检查 conda 环境 `robocup3d` 与关键权重）。

### 运行前配置

1. **修改本机 IP**：编辑主程序第 20 行附近
   ```python
   address = '172.27.246.124'   # 修改为运行程序的电脑 IP
   ```
2. **确保机器人端监听端口**：默认 TCP 端口 `6666`（见 `NetworkThread(host=address, port=6666)`）
3. **连接相机**：优先 Orbbec 深度相机，未检测到自动使用普通摄像头

### 工具脚本

```bash
python camera_test.py          # 测试相机切换（Orbbec ↔ 普通）
python simple_main.py          # 简化版界面演示
python test_imports.py         # 诊断依赖导入问题
python toONNX.py               # 用 weights/0512.onnx 推理 assets/P_image_51.jpg
```

---

## 脚本说明

### 当前主程序

| 脚本 | 说明 |
|---|---|
| `pydect3_2025.py` | v3 主程序：目标检测 + 分割，界面加载、网络通信、比赛主入口 |
| `pydect4_2025.py` | v4 主程序：在 v3 基础上扩展颜色识别（当前与 v3 内容一致，为 4 代入口） |
| `td_recognition.py` | PyQt5 界面定义（美化版，深色渐变主题），被上述主程序 `import` |

### 辅助 / 测试脚本

| 脚本 | 说明 |
|---|---|
| `camera_test.py` | Orbbec 相机与普通摄像头自动切换测试 |
| `simple_main.py` | 简化主程序，仅界面 + 摄像头，便于快速验证环境 |
| `simple_camera_test.py` | 相机切换逻辑测试（无需硬件） |
| `test_imports.py` | 逐模块检查依赖导入，定位 DLL 加载失败原因 |
| `toONNX.py` | ONNX 模型推理演示（含导出 ONNX 的参考代码） |

### 归档脚本（`archive/`）

`archive/` 中保存了历届比赛的完整版本快照（`pydect3_2023.py`、`pydect4_2024.py`、各带日期版本等），仅作历史参考，**不保证开箱即用**。运行归档脚本时需将其引用的权重（如 `det300.pt`、`fruit.pt`、`color_fang_4.pt`）放回脚本同目录，或将路径改为 `../weights/xxx.pt`。

---

## 版本演进

仓库 Git 历史完整记录了演进过程，关键节点：

| 版本 | 提交说明 | 内容 |
|---|---|---|
| 1.x | 初运行 / 环境部署检查 / IP 地址优化 | 2023-2024 基础版：相机测试、界面雏形 |
| 2.0 | 更新 best.pt | 检测模型迭代 |
| 3.x | 启动脚本 + 界面逻辑修改至 2025 标准 | 2025 版界面美化、GPU 注释、模型调整、YOLO11 更新 |
| 4.x | 4.1 第一第二轮初步完成 / 4.2 学长版本更新 | 4 代：颜色识别、比赛版稳定 |

---

## 开发者指南

### 代码结构

主程序采用**多线程**架构（详见 `pydect3_2025.py`）：

| 组件 | 作用 |
|---|---|
| `ModelLoaderThread` | 后台加载检测/分割模型，避免启动卡顿 |
| `CameraThread` | 视频帧采集 + 检测循环，发送结果信号 |
| `NetworkThread` | Socket 连接与数据发送（`pack_send()` 打包协议） |
| `Predictor` | 分割推理封装（返回掩码，含置信度与中心点逻辑） |

信号（`pyqtSignal`）驱动界面更新：`update_image`、`update_label`、`update_result`、`detection_finished`、`connection_ready` 等。

### 常用修改点

| 需求 | 修改位置 |
|---|---|
| 修改通信 IP / 端口 | 主程序 `address` 变量、`NetworkThread(host=..., port=6666)` |
| 切换检测模型 | `YOLO('weights/best.pt')` → 改为 `weights/` 下其他检测权重 |
| 切换分割模型 | `Predictor('weights/yuan0517.pt')` → 其他分割权重 |
| 启用 GPU | `GPU_DEVICE = False` → `True`（需安装 CUDA 版 PyTorch） |
| 修改检测类别 | 代码头部 `data` 字典、`rgb_dict`、`elseObject` |
| 修改检测阈值 | `SAT_NUM`、`Predictor.predict()` 中 `conf=` 参数 |

### 训练新模型

项目使用 ultralytics（YOLOv8 系列 API）训练与导出：

```python
from ultralytics import YOLO

# 训练（以检测为例）
model = YOLO('yolov8s.pt')            # 或官方其他预训练权重
model.train(data='your_dataset.yaml', epochs=100, imgsz=640)

# 验证
model.val()

# 导出 ONNX（部署用）
model.export(format='onnx', simplify=True)
```

训练产出（`best.pt` / `last.pt` 等）统一放入 `weights/` 并在 `weights/README.md` 登记。

### 新增 / 修改界面

界面定义在 `td_recognition.py` 的 `Ui_MainWindow` 类中，直接编辑样式表（`setStyleSheet`）或控件布局即可；美化细节参考 `docs/界面美化详细说明.md`。

### 提交规范

- 权重等大文件**不入库**（已被 `.gitignore` 忽略）；确需分发走 Releases / 网盘
- 提交信息建议遵循既有风格：`X.Y 简要说明`（如 `4.3 修复断线重连`）
- 新脚本放入 `archive/` 前请在 `archive/README.md` 登记版本说明

---

## 常见问题（FAQ）

**Q1：启动报 `No module named 'pyorbbecsdk'`**
未安装 Orbbec SDK，执行 `pip install pyorbbecsdk`；不使用深度相机时程序会自动回退普通摄像头，也可注释相关导入。

**Q2：报 `DLL load failed` / `libOrbbecSDK.so` 找不到**
动态库未在系统路径中。参考[安装部署](#3-安装-orbbec-sdk使用深度相机时)将 `3rdparty/orbbec_sdk/` 对应平台库加入 `PATH` / `LD_LIBRARY_PATH`。可用 `python test_imports.py` 定位具体模块。

**Q3：模型文件找不到（`FileNotFoundError: weights/xxx.pt`）**
未放置权重。请按[准备权重](#4-准备权重)从网盘 / Releases 获取，并确认从**仓库根目录**运行程序（权重路径为相对路径）。

**Q4：检测无画面 / 相机无法打开**
先运行 `python camera_test.py` 查看相机切换逻辑；确认 Orbbec 相机已连接且驱动正常，否则程序会回退到普通摄像头（索引 0）。

**Q5：能否在无 GPU 的机器上运行？**
可以。默认 `GPU_DEVICE = False`，CPU 模式运行；检测速度取决于模型大小，可选 `weights/` 中小模型（如 `det100.pt`、`0512.pt`）或使用 NanoDet 子项目。

**Q6：界面显示但不检测？**
确认检测模型加载成功（状态栏提示），并检查 `YOLO()` 权重路径与 `conf` 阈值（`SAT_NUM`）；摄像头画面正常后点击"重新检测"按钮触发。

---

## 致谢

- [ultralytics / YOLO](https://github.com/ultralytics/ultralytics) —— 检测、分割与训练框架
- [NanoDet](https://github.com/RangiLyu/nanodet) —— 轻量检测子项目（`NanoDet/` 内自带致谢说明）
- Orbbec（奥比中光）—— 深度相机与 SDK
- 历届参赛队员的持续迭代与维护

---

*维护：CUG RoboCup 3D 团队 · 最近更新：2025-09*
