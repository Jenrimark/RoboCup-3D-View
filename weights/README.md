# weights/ — 模型权重清单

本目录集中存放项目全部模型权重（`.pt` / `.onnx`）。

> **重要**：权重为训练产物且体积较大（合计约 **523 MB**），**不入库管理**（见根目录 `.gitignore`）。
> 从共享网盘 / GitHub Releases / 本地备份获取后放入本目录即可；克隆仓库后本目录默认为空。
>
> 运行主程序**至少需要**：`best.pt`（主检测）+ `yuan0517.pt`（分割）。

---

## 当前版本使用的权重

| 文件 | 大小 | 类型 | 使用位置 | 说明 |
|---|---|---|---|---|
| `best.pt` | 5.2 MB | YOLO 检测 | `pydect3_2025.py` / `pydect4_2025.py` | 主检测模型（当前比赛入口） |
| `yuan0517.pt` | 19 MB | YOLO 分割 | `pydect3_2025.py` / `pydect4_2025.py` | 分割模型（返回掩码，`Predictor` 使用） |

## 历史版本使用的权重

| 文件 | 大小 | 类型 | 使用位置 | 说明 |
|---|---|---|---|---|
| `best-detect.pt` | 6.0 MB | YOLO 检测 | `archive/pydect3_2023.py` | 2023 版检测模型 |
| `best-seg.pt` | 6.5 MB | YOLO 分割 | `archive/pydect3_2023.py` | 2023 版分割模型 |
| `det300.pt` | 22 MB | YOLO 检测 | `archive/pydect3_2024*`、`pydect3_2025.7.*` | 2024-2025 主检测模型 |
| `fruit.pt` | 6.0 MB | YOLO 检测 | `archive/pydect3_2024*`、`pydect3_2025.7.*` 等 | 水果检测模型 |
| `best518s.pt` | 43 MB | YOLO 检测 | `archive/pydect4_2024.py`、`pydect4_2025.7.*` | 4 代主检测模型（0518） |
| `final0518seg.pt` | 6.5 MB | YOLO 分割 | `archive/pydect4_2024.py`、`pydect4_2025.7.*` | 4 代分割模型（0518） |
| `color_fang_4.pt` | 6.5 MB | 颜色分类 | `archive/pydect4_2024*` | 方形目标颜色分类 |
| `0512.pt` | 6.0 MB | YOLO 检测 | `toONNX.py`（导出源模型） | 导出 `0512.onnx` 的检测模型 |
| `0512.onnx` | 12 MB | ONNX 推理 | `toONNX.py` | ONNX 格式推理演示 |

## 实验 / 备份权重（当前无脚本直接引用）

> 下列文件为训练过程各阶段产物或实验模型，保留备查；如需使用，请自行确认效果并在 `archive/` 对应脚本中替换引用。

| 文件 | 大小 | 推测用途 |
|---|---|---|
| `aug1.pt` | 68 MB | 数据增强训练版本（大模型） |
| `0514_best.pt` | 22 MB | 0514 训练最优权重 |
| `det100.pt` | 21 MB | 检测模型（100 类配置实验） |
| `st300.pt` | 22 MB | 检测模型（300 步训练实验） |
| `seg0514_1.pt` | 23 MB | 0514 分割实验（第 1 版） |
| `seg0514_5.pt` | 68 MB | 0514 分割实验（第 5 版） |
| `seg0514_6.pt` | 23 MB | 0514 分割实验（第 6 版） |
| `seg0514_7.pt` | 23 MB | 0514 分割实验（第 7 版） |
| `fang_final.pt` | 23 MB | 方形目标最终版 |
| `color_fang_1/2/3.pt` | 19 / 6.5 / 6.5 MB | 方形颜色分类（1-3 号） |
| `color_yuan_1~4.pt` | 6.5 MB ×4 | 圆形颜色分类（1-4 号） |
| `color-seg.pt` | 6.5 MB | 颜色分割模型 |
| `depth-seg.pt` | 6.5 MB | 深度分割模型 |
| `hard_best.pt` | 6.0 MB | 困难样本训练最优 |
| `last.pt` | 6.0 MB | 训练末轮权重 |
| `newbest.pt` | 5.2 MB | 更新版最优（对应 `pydect3_2025_zqy.py` 引用的 `new_best.pt`） |
| `unknown0513.pt` | 6.0 MB | 0513 实验权重 |

---

## 如何新增权重

训练 / 导出完成后：

1. 将权重文件放入本目录（建议命名 `类型_日期.pt`，如 `det_20250911.pt`）
2. 更新本清单（用途、大小、对应脚本）
3. 修改主程序中的加载路径（`YOLO('weights/你的模型.pt')`）
4. 权重分发走共享网盘 / GitHub Releases，**不要提交到 Git 仓库**

## 如何用 Git LFS 替代（可选）

如希望权重也随仓库分发，可用 [Git LFS](https://git-lfs.com) 管理 `.pt` 文件（需将 `.gitignore` 中 `*.pt` 规则移除并配置 `git lfs track`），但注意 GitHub 免费仓库 LFS 配额有限（1 GB 存储 / 1 GB 月流量）。
