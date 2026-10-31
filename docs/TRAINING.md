# AutoDL 训练、评估与推理

本次交付训练程序与模型实现，默认512输入、40轮、microbatch1、梯度累积8、19通道独立多标签分割。完整GPU训练、真实预训练权重下载和论文指标尚未在AutoDL执行；本地检查使用小型随机初始化骨干和CPU。

## 先在云端准备

将 `dacl10k_clean_autodl.7z` 与 `.7z.sha256` 上传到 `/root/autodl-tmp/`，按[数据包说明](PACKAGE.md)校验并解压：

```bash
cd /root/autodl-tmp
sha256sum -c dacl10k_clean_autodl.7z.sha256
7z x dacl10k_clean_autodl.7z -o/root/autodl-tmp
git clone https://github.com/ykkxxx/bridgeDisease.git /root/autodl-tmp/bridgeDisease
cd /root/autodl-tmp/bridgeDisease
```

仓库已克隆时，在仓库目录执行 `git pull --ff-only`。Ubuntu若没有 `7z`，可以安装 `p7zip-full` 后再解压。

镜像已有PyTorch2.1.0/torchvision0.16.0时保留该组合，先检查：

```bash
python -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.cuda.is_available())"
python -m pip install -r requirements-train.txt
python -m pip install -e . --no-deps
export HF_HOME=/root/autodl-tmp/huggingface-cache
```

`requirements-train.txt` 不会自动替换torch。若镜像缺少兼容版本，官方CUDA12.1组合为：

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

依据：[PyTorch旧版本安装说明](https://pytorch.org/get-started/previous-versions/)。

第一次构建模型会下载公开的CLIP、MiT或ResNet预训练骨干，后续复用缓存；7z数据包不含这些模型权重。也可以将 `clip_model_name`、`segformer_model_name` 改成本地HuggingFace模型目录。固定文本特征会在构建时计算并缓存。

默认配置已对应解压后的路径：

```text
data_root=/root/autodl-tmp/bridge_disease_data/dataset
prepared_root=/root/autodl-tmp/bridge_disease_data/prepared
```

## 已提供哪些模型

所有实现位于 [models.py](../src/bridge_seg/models.py)，各配置位于 `configs/experiments/`。

| ID | 实现 | 作用 |
| --- | --- | --- |
| B1 | DeepLabV3+式ASPP+低层融合，ImageNet ResNet50 | CNN基线；小batch时骨干BN统计固定 |
| B2 | ImageNet MiT-B2骨干+随机初始化SegFormer分割头 | 本月强视觉基线 |
| B3 | 与P相同结构，使用可学习19×3视觉类别原型 | 分离CLIP视觉预训练与文本语义收益 |
| B4 | HuggingFace CLIPSeg解码器，多标签适配 | CLIPSeg参照；只加载CLIP骨干，解码器随机初始化 |
| P | BridgeTextSeg：四层CLIP、类内自适应文本匹配、局部CNN、边界监督 | 你的完整模型 |
| A0 | 类别名称、固定平均、无局部/边界 | 基础文本匹配 |
| A1 | 领域描述、固定平均、无局部/边界 | 描述作用 |
| A2 | 领域描述、自适应、无局部/边界 | 去局部细节 |
| A3 | 类别名称、固定平均、有局部、无边界 | 类别名文本对照 |
| A4 | 领域描述、自适应、有局部、无边界 | 去边界监督 |
| A1L | 领域描述、固定平均、有局部、无边界 | 自适应加权对照 |

`A5` 是P的构建别名，不另建配置或重复训练。B5/CAT-Seg属于完整研究范围，未纳入这次本月实现；程序会明确拒绝该ID。

P/B3取CLIP第3/6/9/12层空间特征并插值位置编码。文本分支冻结；图像骨干最后两层在第6轮开始解冻。B4视觉特征每张图共享一次，按类解码并使用激活检查点降低显存，保留原生FiLM条件嵌入尺度。B4使用HuggingFace实现的随机初始化解码器，不使用PhraseCut等分割训练过的检查点，不宣称原论文分数复现。

实现接口核对来源：[CLIP源码](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/clip/modeling_clip.py)、[CLIPSeg源码](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/clipseg/modeling_clipseg.py)、[MiT-B2骨干](https://huggingface.co/nvidia/mit-b2)。

## 第一步：内部开发训练

先打印配置，确认路径和40轮默认值，不下载权重：

```bash
python train.py --config configs/experiments/P.json --dry-run
```

先跑你的模型：

```bash
python train.py --config configs/experiments/P.json --stage dev --device cuda
```

跑SegFormer基线：

```bash
python train.py --config configs/experiments/B2.json --stage dev --device cuda
```

`dev`只读取6,186张内部训练集与687张内部验证集。训练数据按1536基础最长边及随机尺度缩放、随机512裁块/补边和翻转，图片与19通道mask同步变换；重叠标签保留，补边不进入损失。各骨干使用匹配其预训练来源的归一化。

内部验证按完整原图滑窗评分，窗口512、步长352、基础最长边1536；重叠区平均logits，恢复原尺寸后累计TP/FP/FN，再计算各类指标。验证可能占用一定时间；可设 `--val-every 2` 每两轮检查一次。

可以覆盖轮数或启用内部早停：

```bash
python train.py --config configs/experiments/P.json --stage dev --epochs 40 --early-stop-patience 8
```

`--early-stop-patience`以距离最佳内部验证轮次的epoch数计。默认0表示关闭。若验证间隔较大，应相应增大patience。

首次只验证环境时，可以限制批次数与验证图数；此时配置和指标会标记为partial，不能作为完整实验结果：

```bash
python train.py --config configs/experiments/P.json --run-dir outputs/smoke/P \
  --epochs 1 --max-train-batches 2 --max-val-images 1 --num-workers 0
```

## 40轮还是80轮

你报告过往实验约40轮收敛，因此当前默认改为40轮。80轮不是必须遵守的要求。相同条件下减少轮数可减少训练开销，但新模型、解冻策略和学习率曲线可能改变收敛速度，不能据旧实验保证新模型第40轮必然收敛。

先在内部验证集看mIoU、训练损失与最近若干轮的变化。如果40轮前已稳定，不必强行补到80；如果仍持续改善，再在正式测试前将统一预算调整为60或80。内部可以早停和保存best，正式重训应事先锁定所有方法的训练长度和选模规则。不要看966张最终测试集的分数决定继续训练多久。

## 第二步：锁定方案后正式重训

```bash
python train.py --config configs/experiments/P.json --stage final --epochs 40 --seed 42
```

`final`从预训练骨干重新初始化，使用6,873张train+val并集训练；不读取val或test，不按最终测试选checkpoint。最终保存 `final.pt`，此阶段拒绝早停。不要将 `dev` 的优化器状态拿来 `--resume` 到final。

## 第三步：最终测试

```bash
python evaluate.py --checkpoint outputs/final/P_seed42/final.pt \
  --split test --output outputs/final/P_seed42/test_metrics.json --device cuda
```

输出总体mIoU13/mIoU19、各类IoU/Dice/Precision/Recall与TP/FP/FN，同时导出CSV。指标取值0–1；论文展示百分比时乘100。966张是官方validation非空子集，不能写成完整975张。`testdev`无本地真值，不提供它的本地评分入口。

## 中断后继续

```bash
python train.py --config configs/experiments/P.json --stage dev \
  --resume outputs/dev/P_seed42/last.pt
```

保留相同阶段、模型、seed、输入尺寸、总轮数及训练协议。程序核对损失/优化器设置、数据清单、清洗标注、提示和冻结骨干哈希；更换数据存放路径可以，但内容必须一致。冒烟运行的批次数/图数上限也不能在恢复时清除，从而避免把部分训练包装成完整实验。`last.pt`包括优化器、AMP scaler、已完成轮数、Python/NumPy/Torch/CUDA及DataLoader随机状态；每轮完成时原子保存。中途断开时从最后完整轮次继续，未完成的一轮重新运行。

正常预训练模型的检查点只保存可训练参数、候选解冻层和必要buffer，避免每个实验重复保存完整冻结CLIP。重载时仍需要同一预训练骨干来源或缓存，程序会比对其内容哈希；随机初始化的调试配置则保存完整模型。输出目录已有内容时不会静默覆盖，必须换目录或显式 `--resume`。

## 推理并导出19类mask

```bash
python infer.py --checkpoint outputs/final/P_seed42/final.pt \
  --image /root/autodl-tmp/bridge_disease_data/dataset/data/data_79 \
  --output outputs/predictions/P --device cuda
```

也可将 `--image` 指向单张图片。对每张图片生成19个独立二值PNG及预测说明，允许不同类别在同一像素重叠。默认阈值0.5；如需修改，只能依据内部验证确定。

## 批量执行本月核心矩阵

先查看命令：

```bash
python scripts/run_experiments.py --stage final --dry-run
```

确认后顺序运行六模型×三个seed，共18次：

```bash
python scripts/run_experiments.py --stage final --epochs 40
```

建议先全部模型seed42确认可运行，再增加17和2026：

```bash
python scripts/run_experiments.py --stage dev --seeds 42
```

批量脚本遇到非零退出即停止，便于保留错误现场；默认不自动覆盖或跳过已有运行。保存报错末尾、启动命令和 `config.json`，后续可据此修正。

## 输出与检查

论文曲线/逐类表/多seed统计的导出，以及基线补跑和备份步骤见[实验记录说明](EXPERIMENT_RECORDS.md)。现有训练记录已包含每轮与逐类指标，无需为CSV导出修改正在运行的训练程序。

每个运行目录包含 `config.json`、`metadata.json`、`history.jsonl`、`last.pt`；dev另有 `best.pt`，final结束后有 `final.pt`。元数据记录Git提交、数据清单和提示文件哈希、版本与参数量，CUDA训练记录显存峰值。`history.jsonl`内包含每轮训练损失、耗时和内部验证指标。

本地或云端安装开发依赖后可运行小样本测试，无需下载真实预训练权重：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
