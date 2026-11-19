# YOLO26n-OBB 主动学习回流实验计划

版本 v1.0 · 2026-11-19 · 实验方案，尚无训练或回流结果

## 1. 要验证什么

在相同数据池和回流预算下，YOLO26n-OBB 的 one2one 与 one2many 双头分歧，是否比现有低置信度规则多发现独立的漏检、误检、错分或定位错误？将这些图像加入训练后，能否在相近的误报水平下提高独立测试集上的异物召回？

本库有七个类别：`coal`、`rockbolt`、`mesh`、`stick`、`board`、`cap`、`bottle`。**主任务定义为煤流 OBB 异物检测。** 目前没有 `gangue` 标注，实验结果不能用于判断煤矸石二分类或实体分选性能。整个实验先做一次离线回流迭代。

## 2. 冻结数据与实验环境

| 集合 | 图片 | OBB 实例 | 实验用途 |
|---|---:|---:|---|
| L0 | 1,577 | 38,435 | 从通用预训练权重微调出初始模型 W0 |
| U | 1,878 | 46,650 | 各策略使用相同图片池，选样完成后再开放对应标签 |
| V | 729 | 17,071 | 选 W0 权重、校准分数与匹配条件、锁定部署工作点 |
| T | 727 | 19,224 | 策略、预算和评估口径冻结后，统一测试 |

数据根目录为 `/root/autodl-tmp/dataset`。此前云端只读核对确认图片、标签和清单与本机一致，且原 YAML 仍包含 Windows 路径。L0、U、V、T 由采集日期分组；无同日跨集合或完全相同图像跨集合。没有轨迹 ID，应继续对候选做近重复控制。U 的标签在 `U_oracle_labels/labels`，目录隔离为实验约定，由流程保证打分阶段只读取 `U_replay_pool/images`。

环境基线为 AutoDL RTX 3080 Ti（12 GB）、`torch 2.8.0+cu128`、`ultralytics 8.4.115`、`yolo26-autodl` 虚拟环境。CUDA 张量计算已通过；OBB 模型加载、数据加载、训练和回流仍须分别验证。用通用 `yolo26n-obb.pt` 初始化，避免使用可能见过本版 V、T 或 U 的历史 Dataset4911 权重。保存预训练文件 SHA-256、版本和运行参数。

## 3. 准备服务器配置并跑通训练

以下命令用于开始实验时执行。它们生成单独的 Linux 配置，原上传的 YAML 保留。先将官方 `yolo26n-obb.pt` 放到命令中的 `weights` 目录；若已有相同权重在其他位置，可设置 `WEIGHTS` 指向该文件。

```bash
source "$HOME/.venvs/yolo26-autodl/bin/activate"
cd /root/autodl-tmp
export YOLO_CONFIG_DIR="$PWD/yolo26-config"
DATA="$PWD/dataset"
RUNS="$PWD/runs/yolo26_obb_al"
WEIGHTS="$PWD/weights/yolo26n-obb.pt"
mkdir -p "$YOLO_CONFIG_DIR" "$RUNS"
SRC="$DATA/dataset4911_yolo26obb_al.yaml"
CFG="$YOLO_CONFIG_DIR/dataset4911_yolo26obb_al_linux.yaml"
sed "s|^path:.*|path: $DATA|" "$SRC" > "$CFG"
test -s "$WEIGHTS" && sha256sum "$WEIGHTS"
```

依次运行以下检查，每一步通过后再执行下一步。

1. **单图推理。** 从 L0 取一张图，确认日志使用 CUDA、模型任务为 OBB，并能完成预测输出。通用权重在皮带图上可能没有有效检测，保存成功即可通过本项。

   ```bash
   yolo obb predict model="$WEIGHTS" \
     source="$DATA/L0_initial_train/images/d4911_l0_000001.jpg" \
     device=0 imgsz=640 project="$RUNS" name=smoke_predict save=True
   ```

2. **一轮小训练。** 以 5% 的 L0 训练子集验证标签读取、OBB loss、反向传播、V 验证和权重保存。V 仍使用完整验证集。通过标准是无缺失标签或 NaN、训练完整结束并产生权重。训练缓存和输出留在云盘。确认显存有余量后再决定正式 batch。

   ```bash
   yolo obb train model="$WEIGHTS" data="$CFG" \
     device=0 imgsz=640 batch=4 workers=2 epochs=1 fraction=0.05 \
     seed=26 project="$RUNS" name=smoke_train
   ```

3. **训练 W0。** 从同一份通用权重重新初始化，使用完整 L0 和 V。一轮小训练权重仅用于连通性检查。下列参数作为起始配置，最终以 V 和实际显存为准，确定后在实验记录中冻结。

   ```bash
   yolo obb train model="$WEIGHTS" data="$CFG" \
     device=0 imgsz=640 batch=4 workers=2 epochs=100 patience=20 \
     seed=26 project="$RUNS" name=w0_initial
   ```

   W0 默认取 `w0_initial/weights/best.pt`。记录最佳轮次、逐类验证结果及典型失败案例。确认模型在主要类别上给出可分析的预测，再进入回流实验。

## 4. 第一阶段：冻结 W0，比较难例筛选

最小启动版做 R、L、D 三组。若资源允许，加 M 作分类歧义消融。

| 组 | 方法 | 选样规则 |
|---|---|---|
| R | 随机抽样 | 对完整 U 按固定种子抽样，保留抽到的无框帧 |
| L | 低置信度 | 沿用当前规则，事先记录使用的检测头、置信度范围和整帧排序方式 |
| D | 双头分歧 | 同一未融合 OBB 模型计算 one2one / one2many。两头用旋转框相似度匹配，记录类别冲突、单侧出框、定位差异 |
| M（可选） | 类别差值 | 在 one2one 全图 Top-K 之前取候选位置的类别得分。先过滤背景低分，再按同位置前两类分差从小到大排序 |

### 4.1 双头分歧的最小实现

1. 先在 V 中取约 20～30 张有代表性的图，使用未融合 PyTorch W0，以相同预处理做一次双头前向。核对两套原始输出都包含 `boxes`、`scores`、`angle`。自动融合的推理入口可能移除 one2many，须明确检查。不能直接比较两个分支里同索引的候选位置。
2. 分别把两个分支解码为同一坐标系的 `xywhr`，记录使用的置信度门槛、输出上限和 one2many 旋转框去重设置。校准时检查单侧结果是否主要来自过宽的候选门槛或未清除的重复框。
3. 按 `batch_probiou` 做不看类别的一对一框匹配，再把匹配结果分为类别冲突、低重叠和单侧检出。ProbIoU 属于旋转框相似度，记录为 ProbIoU。匹配异常或未匹配的候选不可直接认定为真实漏检。定位分歧用 `1−ProbIoU` 对已匹配框排序，角度差只作可视化复核，避免方向等价和近方形目标造成假信号。
4. 在 V 上定好门槛、每类触发配额、随机种子和同帧排序规则，留存配置后对完整 U 打分。优先把 `frame_id`、`rank`、`reason`、两头类别与分数、旋转框和 ProbIoU 写入结果，不读取 U 的预标注。数据和逐帧结果留在私有数据盘。

### 4.2 人审与判定

- 首轮建议每组审 50 张去重帧。对各组使用一致的去重规则，同一物块或连续重复事件只计一次。同一帧被多个方法选中时，审一次并保留各组成员关系。
- 人审在揭开选择规则之前先对照原图和已有标注，全图核对漏检、误检、错分及定位错误。对贴边、遮挡、标签争议或无法判读的目标单独计数。报告相同图片预算的结果，也记录人工分钟数。
- 主指标为每小时确认的独立错例数。报告 D 比 L 多发现的独立高分错误或单侧漏检、各触发类型分别有多少被确认，以及与 R 和 L 的选样重合率。
- 通过条件：双头结果可复现，且 D 找到的额外错例具有明确可复核的实例。若收益主要来自重复框或不可靠匹配，回到 V 修正规则，再用未参与修订的候选进行复核。

## 5. 第二阶段：相同预算回训

1. 冻结第一阶段有代表性的回流策略和训练配置。为每组分别生成 `L0 + 本组入选 U` 的训练清单，只开放入选样本的 OBB 标签。各组按相同新增图片数做主比较，同时报告目标数与人工耗时。另有人工标注计时数据时，可增加等人工时间比较。
2. R、L、D 各组都独立从同一个 W0 微调一次。保持优化器、学习率、训练步数、输入尺寸、增强和随机种子一致，避免逐组沿用上一组模型或发生预算混淆。保存每组入选清单、参数与权重哈希。V 用于选择预先规定的 checkpoint 和工作点。
3. T 只用于统一的最终评估。部署评估固定同一个推理分支和后处理设置，以 V 上确定的相近误报工作点评估各组。报告 T 上实际误报、关键异物召回率、逐类 AP、整体 OBB mAP50–95 和独立错误数。优先同时看 `rockbolt`、`mesh` 等有相对更多实例的类别。T 无 `bottle`，不报告其测试效果；`cap` 例数很少，指标仅作观察。
4. 结论按证据范围陈述。若 D 多找错例且在相近误报水平下提高召回，可进入下一轮回流。若只改善选样，先保留为人工复核入口。首轮差异小时，重复训练种子并按采集批次报告变化，避免把随机波动解释成稳定收益。

## 6. 留档、提交与边界

- 每轮留档：环境与 checkpoint 哈希、数据划分版本、采样分组、阈值、入选清单、标注时间、训练配置、逐类指标和已确认的错例类型。GitHub 只备份可分享的实验文档、代码与通用配置，使用中文提交记录。
- 图片、标签、回流清单、权重、预测文件、运行日志和 SSH 凭据留在私有存储。用于回流训练的数据路径只在服务器本地配置里解析。
- 参考实现与方法：[Ultralytics OBB 任务文档](https://docs.ultralytics.com/tasks/obb)、[Ultralytics 双头推理说明](https://docs.ultralytics.com/guides/end2end-detection)、[PPAL，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Plug_and_Play_Active_Learning_for_Object_Detection_CVPR_2024_paper.html)。双头分歧用作本数据的筛选指标属于待验证方案。
