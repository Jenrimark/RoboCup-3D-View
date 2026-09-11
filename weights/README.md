# weights/ — 模型权重

本目录仅保留**当前正式版代码使用的权重**（约 24 MB）。

> **重要**：权重为训练产物，**不入库管理**（见根目录 `.gitignore`）。
> 2026-09 已清理历史/实验权重（原 33 个 → 2 个），旧权重仍可通过 Git 历史或团队网盘找回。

---

## 当前使用的权重

| 文件 | 大小 | 类型 | 使用位置 | 说明 |
|---|---|---|---|---|
| `best.pt` | 5.2 MB | YOLO 检测 | `src/pydect3_2025.py` / `src/pydect4_2025.py` | 主检测模型（当前比赛入口） |
| `yuan0517.pt` | 19 MB | YOLO 分割 | `src/pydect3_2025.py` / `src/pydect4_2025.py` | 分割模型（返回掩码，`Predictor` 使用） |

## 已清理的历史/实验权重

以下权重曾存放于本目录，已于 2026-09 清理（如需可联系团队网盘或从 Git 历史恢复）：

| 文件 | 原用途 |
|---|---|
| `best-detect.pt`、`best-seg.pt` | 2023 版检测 / 分割 |
| `det300.pt`、`fruit.pt`、`yuan0517.pt`（旧版） | 2024 版主检测 / 水果检测 |
| `best518s.pt`、`final0518seg.pt` | 4 代 0518 检测 / 分割 |
| `color_fang_1~4.pt`、`color_yuan_1~4.pt`、`color-seg.pt` | 颜色分类 / 颜色分割 |
| `0512.pt`、`0512.onnx` | ONNX 导出源模型 / 推理演示 |
| `aug1.pt`、`0514_best.pt`、`det100.pt`、`st300.pt`、`seg0514_1/5/6/7.pt`、`fang_final.pt`、`hard_best.pt`、`last.pt`、`newbest.pt`、`unknown0513.pt`、`depth-seg.pt` | 训练过程各阶段产物 / 实验模型 |

> 归档脚本（`archive/`）若需运行，请将对应权重复制到 `weights/` 后再改脚本引用。

---

## 如何新增权重

1. 训练 / 导出完成后放入本目录（建议命名 `类型_日期.pt`）
2. 修改主程序加载路径（`YOLO('weights/你的模型.pt')`）
3. 更新本清单并同步分发到团队网盘 / GitHub Releases（**不要提交到 Git 仓库**）
