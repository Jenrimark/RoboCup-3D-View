# archive/ — 历史版本脚本存档

本目录保存历届比赛（2023-2025）的开发版本快照，用于**历史参考与回溯**。当前维护版本见 `src/pydect3_2025.py` / `src/pydect4_2025.py`。

> ⚠️ **不保证开箱即用**：归档脚本引用的权重（如 `det300.pt`、`fruit.pt`、`color_fang_4.pt` 等）现位于 `../weights/`，且界面依赖 `../src/td_recognition.py`。如需运行归档版本，请将所需权重复制到脚本同目录，或把脚本中的路径改为 `../weights/xxx.pt`，并从仓库根目录运行。

---

## 版本清单

### v3 系列（检测 + 分割）

| 文件 | 阶段 | 引用权重 | 说明 |
|---|---|---|---|
| `pydect3_2023.py` | 2023 版 | `best-detect.pt`、`best-seg.pt`、NanoDet `model_best.pth` | 最早版本，含 NanoDet 联动 |
| `pydect3_2024.py` | 2024 版 | `det300.pt`、`yuan0517.pt`、`fruit.pt` | 加入水果检测模型 |
| `pydect3_2024 (copy).py` | 2024 备份 | 同上 | 与 2024 版同内容备份 |
| `pydect3_2025_zqy.py` | 2025 早期 | `det300.pt`、`yuan0517.pt`、`fruit.pt`、`new_best.pt` | 张青云同学版本 |
| `pydect3_2025.7.20.py` | 2025-07-20 | `det300.pt`、`yuan0517.pt`、`fruit.pt` | 2025 开发快照 |
| `pydect3_2025.7.26.py` | 2025-07-26 | 同上 | 2025 开发快照（加载状态优化） |
| `pydect3_2025.7.27.py` | 2025-07-27 | `best.pt`、`yuan0517.pt`、`fruit.pt` | 切换到 `best.pt` 主模型 |

### v4 系列（检测 + 分割 + 颜色识别）

| 文件 | 阶段 | 引用权重 | 说明 |
|---|---|---|---|
| `pydect4_2024.py` | 2024 版 | `best518s.pt`、`final0518seg.pt`、`fruit.pt` | 4 代起点，脚本目录定位权重（`os.path.join`） |
| `pydect4_2024 (copy).py` | 2024 备份 | `det300.pt`、`yuan0517.pt`、`color_fang_4.pt`、`fruit.pt` | 另一分支：颜色识别 |
| `pydect4_2024-0518.py` | 2024-05-18 | 同上 | 0518 快照 |
| `pydect4_2025.7.20.py` | 2025-07-20 | `best518s.pt`、`final0518seg.pt`、`fruit.pt`、`yolov5s.pt` | 4 代 2025 快照 |
| `pydect4_2025.7.26.py` | 2025-07-26 | 同上 | 4 代 2025 快照 |
| `pydect4_2025.7.27.py` | 2025-07-27 | `best.pt`、`yuan0517.pt`、`fruit.pt` | 切换到 `best.pt` 主模型 |

---

## 归档规范

- 新增历史版本时，请在本表登记（文件名、阶段、引用权重、说明）
- 脚本中引用的权重请一并登记到 `../weights/README.md`
- 归档脚本**不要**再提交权重大文件（已被 `.gitignore` 忽略）
- 完整演进历史也可通过 `git log --oneline` 查看（提交信息格式：`X.Y 说明`）
