# 论文实验记录、导出与基线顺序

当前40轮P运行应保留，无须因基线尚未运行而重跑。前面的P命令先验证了完整链路；后续开发顺序采用B2→B3→B4→P→A2→A4。模型没有必须先训练基线的参数依赖，公平比较依靠相同数据协议、分辨率、预算及评估方式。

## 当前训练已经自动保存什么

默认运行目录是 `outputs/dev/P_seed42/`；若启动时指定 `--run-dir`，以该目录为准。

| 文件 | 用于论文或复现 |
| --- | --- |
| `history.jsonl` | 每轮训练损失及分量、内部验证mIoU13/19、逐类IoU/Dice/Precision/Recall与TP/FP/FN、耗时、显存峰值 |
| `config.json` | 模型ID、seed、40轮预算、学习率、损失权重、数据指纹、是否partial等 |
| `metadata.json` | 代码提交、参数量、PyTorch版本、数据/提示/骨干哈希 |
| `best.pt` | dev阶段内部验证mIoU13最优模型，可用于定性预测 |
| `last.pt` | 恢复训练的参数、优化器、scaler和随机状态 |
| `final.pt` | final阶段按锁定轮数完成的模型；dev不会生成 |

这些文件每轮完成时保存。终端日志只作辅助，不能代替结构化记录。当前程序不保存逐batch曲线，也没有记录实际学习率轨迹或专用推理FPS；这些缺失值不能事后伪造。显存记录为从本次训练开始累计的峰值，不等于每轮单独重置测量。

在服务器另一个终端，可补存当前环境快照，先替换为实际运行目录：

```bash
cd ~/autodl-tmp/bridgeDisease
RUN_DIR=outputs/dev/P_seed42
python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
nvidia-smi > "$RUN_DIR/gpu_snapshot.txt"
```

快照记录采集时刻的环境。训练时长、峰值显存和独立推理速度是不同指标，论文效率表需要另做同硬件、同输入设置的推理计时。

## 无需改动正在训练的程序，导出CSV

`scripts/export_results.py` 只读取已保存记录，不载入模型、更新权重或改写训练目录；不需要Torch。更新代码后，在仓库目录执行：

```bash
python scripts/export_results.py \
  --runs-root outputs \
  --output outputs/paper_tables_$(date +%Y%m%d_%H%M%S)
```

这是当前时刻的只读快照。训练期间可导出已完成epoch；最后一行尚未写完时会跳过并记录警告。训练结束后再导出一次到新目录，程序拒绝覆盖已有导出结果。

| 输出 | 后续用途 |
| --- | --- |
| `curves.csv` | 横轴epoch，绘制loss、内部验证mIoU曲线，比较收敛过程 |
| `run_summary.csv` | 每次运行的完成状态、seed、最佳内部轮次、参数量及训练耗时 |
| `per_class_metrics.csv` | 逐类对比表、热力图、类别IoU柱状图；含原始TP/FP/FN |
| `evaluations.csv` | 独立评估报告的mIoU、样本数、阈值和检查点/清单哈希 |
| `seed_summary.csv` | 相同模型、阶段和协议下的实际seed均值及样本标准差 |
| `export_report.json` | 导出时间、记录数量、警告及缺失/未完成情况 |

指标保持0–1，论文百分比展示时乘100。单seed没有标准差，CSV留空；不冒充3seed。重复seed、未完成或partial训练、部分评估不会被混入多seed统计。不同代码/数据/配置协议不会合并。dev的最佳内部验证与final的最终测试分表统计，不能放在同一列当作同一测试结果。

final阶段完成后，先用[训练说明](TRAINING.md)中的 `evaluate.py` 生成该运行目录内的 `test_metrics.json`。最终汇总只接受唯一、完整、且SHA256对应本目录 `final.pt` 的test报告；不挑多个报告中分数最高的一个。不满足条件时保留原始记录并发出警告。

## 接下来如何补基线

先让当前P运行结束，再顺序启动基线，避免同一张卡同时抢显存与算力。先补B2、B3、B4的seed42内部开发：

```bash
python scripts/run_experiments.py \
  --stage dev --models B2 B3 B4 --seeds 42 --epochs 40
```

当前P结果可与这些运行在同一dev协议下比较。随后补A2、A4：

```bash
python scripts/run_experiments.py \
  --stage dev --models A2 A4 --seeds 42 --epochs 40
```

通过内部验证锁定方案后，再按所有方法相同协议，用final_train重训。本月核心六模型×seed42/17/2026共18次最终训练：

```bash
python scripts/run_experiments.py --stage final --epochs 40
```

最后统一在966张非空官方validation子集评分，注明该口径；不利用它挑选提示、阈值、训练轮数或最优checkpoint。各方法应使用相同的预注册定性样本生成预测图，不按哪个方法效果好来挑不同图片。

## 哪些数据要备份

保留整个实验运行目录、原始history、配置、元数据、评估JSON/CSV、用于展示的预测mask及原图路径清单。云端 `outputs/` 被Git忽略，不会随 `git push` 自动回本地；预训练缓存也应保留，以便重建只保存可训练参数的检查点。

每次训练结束后，可打包实验输出，排除已有5GB数据压缩包与缓存：

```bash
cd ~/autodl-tmp/bridgeDisease
tar --exclude='outputs/huggingface-cache' \
    --exclude='outputs/test-*' --exclude='outputs/pytest-*' \
    --exclude='outputs/*.7z*' \
    -czf /root/autodl-tmp/bridge_experiments_$(date +%Y%m%d_%H%M%S).tar.gz outputs
```

将生成的实验压缩包下载到本地或独立存储。不要等到释放实例时才备份，也不要只保留最终画好的图而删除原始数值记录。
