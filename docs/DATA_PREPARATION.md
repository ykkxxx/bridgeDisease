# 数据清洗、划分与三类集合的作用

入口代码：[scripts/prepare_dataset.py](../scripts/prepare_dataset.py)。本次采用你已选择的现有实验方案：在官方训练集内约90/10划分训练和内部验证，将官方 `validation` 保留为最终测试；官方 `testdev` 无本地真值，单列为推理集。

## 训练集、验证集、测试集分别做什么

| 本项目名称 | 来源 | 作用 | 能否用来调整模型 |
| --- | --- | --- | --- |
| `train`，训练集 | 清洗后官方 train 的约90% | 输入图片与标签，计算损失、反向传播、更新模型参数 | 用于学习参数 |
| `val`，内部验证集 | 清洗后官方 train 的约10% | 开发期间检查泛化、调整学习率/阈值/提示、早停或选择 checkpoint | 可以，但不参与梯度更新 |
| `test`，本地最终测试集 | 清洗后官方 validation | 所有方案和选模规则锁定后，统一计算论文最终指标 | 不可以根据其结果反复调参或挑模型 |

训练集好比日常练习，验证集好比用于改进学习方法的模拟考试，测试集是最后的统一考试。即使程序在验证和测试时都不反向传播，反复根据结果调整模型也会消耗该集合的独立评估作用，因此内部验证和最终测试需要分开。

本项目把官方 `validation` 用作本地最终测试，是研究协议中的角色约定，不代表它改名成了官方隐藏测试集。论文应明确说明来源及具体清洗范围。

另外生成两类清单：

- `final_train`：内部 `train + val` 的并集。用内部实验锁定超参数、训练轮数和 checkpoint 规则之后，用它重新训练正式模型；最终再到 `test` 统一评估。不能根据 `test` 调整重训轮数或挑最好一轮。
- `inference`：原始 `testdev` 中通过图片检查的样本。没有本地分割真值，只能产生预测或用于官方评测提交，不能当全零负样本，也不能计算本地 IoU/Dice。

所有模型使用同一份划分清单和 seed，不能为不同方法重新随机划分；图片裁块和增强在划分之后进行，同一原图产生的块始终沿用原图所属集合。

## 本次实际运行结果

2026-10-29使用原始8,922条记录全量运行，seed=42、内部验证比例0.1、空标注策略exclude：

| 清单 | 图片数 | 备注 |
| --- | ---: | --- |
| train | 6,186 | 内部训练 |
| val | 687 | 内部调参与选模，实际约9.996% |
| test | 966 | 官方 validation 的非空子集 |
| final_train | 6,873 | train 与 val 的并集，不是新增样本 |
| inference | 1,012 | 无本地真值的 testdev |

排除71条显式空标注记录，其中官方train62条、validation9条；本次未发现缺失或不可解码图片。发现2组完全相同的图片并作分组约束；19类在内部训练、内部验证和最终测试中均有样本。坐标审计记录951张图、1,059个区域、5,099个越界顶点，全部原样保留。

独立校验通过：原始标注及先前清洗文件的哈希未改变；训练/验证/最终测试之间没有ID交叉或完全相同图片的跨集合泄漏；全部标注索引有效；23个输出文件的哈希与摘要一致。近重复和桥梁级独立性不在这项校验的保证范围内。

## 清洗做了什么

1. 读取原始 `samples.json`；检查相对图片路径，不允许路径逃逸到数据根目录之外。
2. 实际打开并解码图片，获取真实宽高；缺失或不可解码图片进入排除报告。标注中的尺寸不一致时使用图片实际尺寸，并单独记录，不修改源标注。
3. 检查19类标签、闭合/填充字段、每个轮廓至少三个不同顶点、有限二维坐标等；异常样本进入排除报告。
4. 默认延续已有非空清洗：有监督集合中的 `polylines=[]` 记录移出该子集。显式空标注不等于错误标注，也不等于 `testdev` 的 null；经确认属于有效负样本时，可用 `--empty-policy keep` 保留。
5. 坐标超出 `[0,1]` 时仅记录，不改坐标，也不因此删除整张图。后续生成 mask 时应在图像画布内栅格化；此脚本不生成 mask，不改变多类别重叠关系。
6. 对图片文件计算 SHA-256。相同字节内容的训练图片作为同组进入内部训练或内部验证；若训练图片与最终测试图片完全相同，则移出训练端并记录，保留最终测试端。
7. 按 seed=42 对训练内容组作稳定排序，再用组合计数选择最接近10%图片数的分组；相同条件下结果固定。组不可拆开，因此可达到的比例可能略有偏差；摘要会记录实际比例。跨来源的重复路径或 ID 优先保留有效的官方 validation 记录。
8. 导出清洗后的标注、各集合清单、排除原因、逐类分布及哈希。原图片、原始标注、此前的 `samples_clean.json` 与 `cleaning_report/` 均保留。

脚本只做程序可以复核的完整性检查。没有桥梁ID，不能宣称桥梁级划分独立；内容哈希只能识别完全相同的文件，近重复图片、视觉误标仍需人工核验。当前也没有执行多标签分层抽样，应查看逐类分布后再锁定实验清单。

## 默认非空子集与完整官方评估的区别

此前清洗得到官方 train 的6,873张非空样本、官方 validation 的966张非空样本。脚本会重新检查原始导出，以实际结果为准。

默认 `--empty-policy exclude` 排除62条训练空标注和9条官方验证空标注。若最终测试为966张，论文必须注明“官方验证集的非空标注子集”，不能写成完整官方975张结果。

如果需要包含有效空标注的完整协议，应先核验这些记录，再用 `--empty-policy keep` 生成一个**新的**输出目录，并让所有对比方法统一改用该协议。此模式仍会排除确有完整性问题的样本，因此是否保留全部975张仍以报告为准。

## 输出文件怎么用

默认输出到仓库内的 `data/prepared/`，被 `.gitignore` 忽略，不上传大型标注与清单。

| 文件 | 用途 |
| --- | --- |
| `samples_clean.json` | 清洗后有监督原始记录，仍是 `{"samples": [...]}`；保留源记录字段与多边形 |
| `manifests/train.jsonl` | 内部训练清单 |
| `manifests/val.jsonl` | 内部验证清单 |
| `manifests/test.jsonl` | 来自官方 validation 的最终测试清单 |
| `manifests/final_train.jsonl` | 内部 train+val，供协议锁定后重训 |
| `manifests/inference.jsonl` | 无本地真值的 testdev 推理清单 |
| `manifests/train_inner.jsonl`、`tune_inner.jsonl`、`final_eval.jsonl` | 与已有实验计划对齐的别名，分别等同于 train、val、test；不要重复合并使用 |
| `splits/*.txt` | 对应集合的图片相对路径，方便检查或其他程序导入 |
| `reports/summary.json` | 数量、划分参数、源哈希、各输出哈希、评估口径及限制 |
| `reports/excluded_samples.csv` | 排除样本和原因；没有删除源图片 |
| `reports/coordinate_warnings.csv` | 坐标越界记录，仅报告不改写 |
| `reports/metadata_mismatches.csv` | 图片实际尺寸与原 metadata 不同的记录 |
| `reports/duplicate_groups.csv` | 内容完全相同的图片组及原划分 |
| `reports/class_distribution.csv` | 各集合每类出现的图片数，多标签计数不能相加当图片总数 |
| `preview.json` | 每个集合的一条小型清单样例，无需打开大型原文件 |

JSONL 一行一张图片，主要字段：

- `filepath`：相对原始数据根目录，例如 `data/data_0/xxx.jpg`；AutoDL 只需换数据根目录。
- `source_split`：原官方划分，保留 `train`、`validation` 或 `testdev`，方便核对来源。
- `split`：本项目角色 `train/val/test/final_train/inference`。
- `annotation_available`：是否有可用于有监督任务的标注；推理清单为 false。
- `width/height`：从实际图片读取的尺寸。
- `image_sha256`：图片内容哈希，用于分组和验证。
- `annotation_file` 与 `annotation_index`：相对准备输出根目录的 `samples_clean.json`，以及其 `samples` 数组下标。推理样本两者均为 null。
- `labels`：图片包含的类别名称。空列表与 `annotation_available=false` 应结合判断，不能仅看列表是否为空。

清单不是像素 mask。后续数据加载器要按 `annotation_index` 读取原多边形，再按固定19通道生成独立二值 mask；异类交叉处保留多个正标签。

## Windows 运行

需要 Python3.10+ 和 Pillow。使用自己安装的 Python 时：

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
python -m pip install -r requirements-data.txt
python scripts/prepare_dataset.py --data-root 'D:\dataset\dacl10k' --output 'data/prepared' --seed 42 --val-ratio 0.1 --empty-policy exclude
```

本机系统 `python` 目前是应用占位符，`py` 也未发现已注册解释器。本次验证使用 Codex 附带且已有 Pillow 的解释器，可以这样调用：

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
$pythonExe = 'C:\Users\13797\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $pythonExe scripts/prepare_dataset.py --data-root 'D:\dataset\dacl10k' --output 'data/prepared-rerun' --seed 42 --val-ratio 0.1 --empty-policy exclude
```

`data/prepared/` 是本次生成目录；重跑时请换一个新目录，例如上面的 `data/prepared-rerun/`。脚本拒绝覆盖已有输出，也拒绝写入原始数据目录内部。

## AutoDL 运行

先按 [同步说明](SYNC.md) 获取代码、上传完整原始数据，然后在实例终端运行：

```bash
cd /root/autodl-tmp/bridgeDisease
python -m pip install -r requirements-data.txt
python scripts/prepare_dataset.py \
  --data-root /root/autodl-tmp/datasets/dacl10k \
  --output /root/autodl-tmp/bridgeDisease/data/prepared \
  --seed 42 --val-ratio 0.1 --empty-policy exclude
```

两端使用相同原始图片、标注与参数时，清单内容可复现；摘要内的绝对路径和生成时间会不同。可以比较 `reports/summary.json` 中的 `source_sha256` 与 `artifact_sha256` 中 `manifests/*.jsonl` 的哈希，确认使用同一份划分。这些 JSONL 强制使用 LF 换行；辅助 TXT/JSON 的系统换行可能不同。

## 测试与检查

```bash
python -m unittest discover -s tests -v
python scripts/prepare_dataset.py --help
```

测试使用独立的小图片和临时标注，覆盖空/null标注、损坏/缺失图片、未知类别、非有限坐标、越界保留、重复分组、防泄漏、稳定 seed 和输出保护，不读取或改写完整原始数据集。
