# YOLO 旋转框标注与审核工作台

本项目运行在本机 CPU 上。GPU 服务器负责训练、预测和生成待审核任务。你手动将任务 ZIP 下载到本机，在工作台复核、补标和修正，再将结果 ZIP 手动上传到服务器。

## 在 Windows 启动

使用现有的 `YoloWebAgentStarter` Conda 环境时，在 PowerShell 中运行下面两行。

```powershell
Set-Location 'C:\Users\13480\Desktop\codex_\yolo_biaozhu'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

保持终端运行，按 `Ctrl+C` 可以停止服务。终端显示服务已监听后，打开 [本机标注工作台](http://127.0.0.1:5187/)。

应用通过项目内的 Python 服务运行，默认使用现有 Conda 解释器，也可用 `-PythonPath` 指定另一个现有解释器。数据库和媒体由项目目录内的 `.local` 管理。默认端口为 `5187`，可通过 `-Port` 指定另一个本机端口。独立验收时可用 `-DataDir` 指定另一份数据目录。

## 从 GPU 下载并导入

1. 下载 YOLO OBB 数据集 ZIP 到本机。其中应有 `data.yaml`、图片及同名 OBB 标签，标签一行包括类别编号和四个按序排列的归一化顶点。已有标签会作为初始标注，可从仅含 `val` 的数据集开始审核。
2. 如果服务器另外生成了审核资料，选择该 ZIP 作为伴随包一并导入。现有 W0 包中的 `review_frames.csv` 提供审核顺序，`review_events.csv` 提供模型分歧，`previews/` 提供四窗对照图。同包或伴随包中有 `details.jsonl` 时，可以在编辑页直接查看单路预测框。
3. 核对任务图片数和审核事件数。对于已导入的任务，可从任务列表重新打开并接着审核。

一个只有验证集的 OBB 包可以这样组织。

```text
data.yaml
images/val/d4911_v_000004.jpg
labels/val/d4911_v_000004.txt
```

```yaml
val: images/val
names: [coal, rockbolt, mesh, stick, board, cap, bottle]
```

类别顺序应和训练中使用的配置一致。文件名用于匹配原图、人工标注、服务器候选和审核记录。

在 GPU 端生成单个审核任务包时，可在服务器的 Python 环境中运行下列命令。将参数换成本次运行的文件路径和实际来源分集。

```bash
python -B tools/pack_review.py W0_V24_OBB.zip W0_V24_review_task.zip --review W0_V24_review.zip --predictions details.jsonl --summary summary.json --run-id W0 --split V
```

`--review` 也可以接收已解压的审核资料目录。`--predictions` 提供画布预测层，`--summary` 和来源参数记录权重与运行信息。常规补标任务可直接导入原始 YOLO OBB ZIP。

## 逐图审核

打开一张图后，核对可编辑的已有标注。按需要添加或删除框，修改类别、位置、尺寸和角度。审核双头分歧时，查看相应的预测层和事件，记录漏检、误检、错分、定位偏差、重复框或无法判断。审核完成后保存。标注修改由系统留痕，审核结论由你选定。

有些图的标注本身正确，只需要保存对模型事件的判定和完成状态。确实没有目标的图片可以按界面操作确认。对于原标签中的贴边 OBB 四点，可在本机查看和审核。导出会保留未编辑框的原标签行，人工调整过的框以保存的几何坐标导出。

## 导出并回到 GPU 服务器

在任务内下载结果 ZIP，核对审核进度后上传到 GPU 服务器作为新的审核结果。ZIP 包含 `data.yaml`、图片、YOLO OBB 标签和四份审核资料。

| 审核资料 | 内容 |
|---|---|
| `review/task.json` | 任务信息与来源 |
| `review/frames.csv` | 图片及逐图审核状态 |
| `review/events.csv` | 模型分歧事件与人工判定 |
| `review/changes.jsonl` | 标注操作历史 |

任务带有来源信息时，ZIP 还会包含 `review/source.json`。上传后可以用发送给审核者的原任务包与导出包做一致性检查。

```bash
python -B tools/validate_result.py W0_V24_review_task.zip W0_V24_result.zip --expected-split V --unpack datasets/W0_V24_reviewed_v1
```

校验会核对原图、分集、类别编号和来源，并列出实际变化的标签文件。`--unpack` 在校验通过后创建新版本目录。当前 W0 的 V 集任务用于校准和复测，导出后仍保持 `val` 分集。后续 U 池任务可按实验方案选出已审核的回流样本，并在服务器生成下一轮训练数据版本。
