# bridgeDisease

桥梁病害多标签语义分割研究项目。本地开发目录为 `D:\bridgeDisease`，计划在 AutoDL 上完成数据准备和训练。

当前仓库包含研究方案、数据清洗与划分、基线和BridgeTextSeg模型、训练/评估/推理入口。默认40轮；完整GPU训练和论文指标仍待AutoDL实测。

## 文件放在哪里

| 目录或文件 | 用途 | 提交到 GitHub |
| --- | --- | --- |
| `src/` | 后续模型、数据加载和训练代码 | 是 |
| `scripts/` | 数据清洗、划分及后续训练和评估入口 | 是 |
| `configs/` | 可共享配置、路径示例 | 是，排除 `*.local.*` 和 `configs/local/` |
| `docs/` | 已有论文方案、实验计划和同步说明 | 是 |
| `data/` | 数据放置说明，可在本机存放临时数据 | 仅 README |
| `outputs/` | 训练输出、预测图和临时结果 | 仅 README |
| `checkpoints/`、`weights/`、`logs/` | 权重和运行日志，可按需创建 | 否 |
| `.planning/` | 现有本地工作记录，保留在本机 | 否 |

原始数据集继续放在 `D:\dataset\dacl10k`，位于仓库之外。不要为了提交代码把整份数据复制进仓库。

已保留的研究材料：

- [模型架构与实验方案](docs/thesis/CLIP桥梁病害分割_架构与实验方案.md)
- [30 天实验计划](docs/superpowers/plans/2026-10-29-30-day-experiments.md)
- [GitHub 与 AutoDL 同步步骤](docs/SYNC.md)
- [数据清洗、划分与训练/验证/测试集作用](docs/DATA_PREPARATION.md)
- [AutoDL训练、评估与推理](docs/TRAINING.md)
- [清洗数据7z打包与解压](docs/PACKAGE.md)
- [论文实验数据保留、CSV导出与基线顺序](docs/EXPERIMENT_RECORDS.md)

## 云端快速启动

将 `outputs/dacl10k_clean_autodl.7z` 上传并解压到 `/root/autodl-tmp/` 后，在代码仓库执行：

```bash
python -m pip install -r requirements-train.txt
python -m pip install -e . --no-deps
python train.py --config configs/experiments/P.json --stage dev --device cuda
```

默认路径已对应压缩包目录。支持B1–B4、P及A0/A1/A2/A3/A4/A1L；本月六模型批量入口为 `scripts/run_experiments.py`。完整命令见[训练说明](docs/TRAINING.md)。

## 数据清洗与划分

`scripts/prepare_dataset.py` 读取原始 FiftyOne 标注，检查图片与分割区域，生成可复现的训练、内部验证、最终测试及推理清单，保留所有原始文件。依赖见 `requirements-data.txt`。

采用已选实验协议：官方 train 内部约90/10划分；官方 validation 留作本地最终测试；无真值 testdev 单列为推理集。参数锁定后，使用内部 train+val 的 `final_train` 重训正式模型。

默认输出 `data/prepared/` 并延续既有非空标注子集；966张最终测试子集的结果不能标为完整官方975张结果。运行命令、输出格式及两种空标注处理策略见[数据准备说明](docs/DATA_PREPARATION.md)。

## 同步方式

- 本地代码 → GitHub → AutoDL：使用 Git 提交、推送和拉取。
- 本地数据 → AutoDL、AutoDL 结果 → 本地：使用 SCP 或文件传输工具。
- 数据、权重和运行日志不会随 `git push` 上传，也不会随 `git pull` 下载。SCP 是手动文件拷贝；当前没有配置自动同步任务。

`configs/paths.example.json` 展示两端建议使用的路径。需要本机专用配置时，复制为 `configs/paths.local.json` 后修改；该文件已被忽略。示例目前只是路径约定，尚无训练程序读取它。

## GitHub 仓库

仓库地址：[ykkxxx/bridgeDisease](https://github.com/ykkxxx/bridgeDisease)。本地使用 `main` 分支，`origin` 已配置为此仓库。

检查远程地址并推送：

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
git remote -v
git push -u origin main
```

推送需要 GitHub 的有效登录凭据；Git 提交作者姓名不代表已经登录 GitHub。不要把 Token、密码或私钥写进仓库、远程 URL 或共享配置。

以上步骤依据 [GitHub 官方文档](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 平时提交代码

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
git status
git add .
git diff --cached --stat
git commit -m "Describe the change"
git push
```

每次提交前检查暂存清单。`.gitignore` 只影响未跟踪文件，不能阻止 `git add -f`，也不会自动移除已经跟踪的文件。

AutoDL 首次克隆和后续更新、数据上传与结果下载的完整命令见 [docs/SYNC.md](docs/SYNC.md)。
