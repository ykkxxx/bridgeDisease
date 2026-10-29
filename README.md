# bridgeDisease

桥梁病害多标签语义分割研究项目。本地开发目录为 `D:\bridgeDisease`，计划在 AutoDL 上完成数据准备和训练。

当前仓库包含研究方案、实验计划和项目目录骨架；模型与训练代码尚未实现。

## 文件放在哪里

| 目录或文件 | 用途 | 提交到 GitHub |
| --- | --- | --- |
| `src/` | 后续模型、数据加载和训练代码 | 是 |
| `scripts/` | 后续数据处理、训练和评估入口 | 是 |
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

## 同步方式

- 本地代码 → GitHub → AutoDL：使用 Git 提交、推送和拉取。
- 本地数据 → AutoDL、AutoDL 结果 → 本地：使用 SCP 或文件传输工具。
- 数据、权重和运行日志不会随 `git push` 上传，也不会随 `git pull` 下载。SCP 是手动文件拷贝；当前没有配置自动同步任务。

`configs/paths.example.json` 展示两端建议使用的路径。需要本机专用配置时，复制为 `configs/paths.local.json` 后修改；该文件已被忽略。示例目前只是路径约定，尚无训练程序读取它。

## 首次发布到 GitHub

本地仓库使用 `main` 分支。首次发布前，在 GitHub 建立一个名为 `bridgeDisease` 的空仓库，不要勾选创建 README、`.gitignore` 或 License。然后在本地 PowerShell 中执行：

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
# 替换成你刚创建的真实仓库地址。
$repoUrl = 'https://github.com/YOUR_GITHUB_USERNAME/bridgeDisease.git'
git remote add origin $repoUrl
git remote -v
git push -u origin main
```

推送需要 GitHub 的有效登录凭据；Git 提交作者姓名不代表已经登录 GitHub。不要把 Token、密码或私钥写进仓库、远程 URL 或共享配置。若仓库尚未准备公开，可以先创建为私有仓库。

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
