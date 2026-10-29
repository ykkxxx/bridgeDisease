# 本地、GitHub 与 AutoDL 同步

适用于本地 Windows PowerShell 和 AutoDL Linux 实例。代码由 Git 管理；大数据和训练产物通过文件传输管理。

## 1. 建议的两端目录

| 内容 | 本地 Windows | AutoDL |
| --- | --- | --- |
| Git 代码仓库 | `D:\bridgeDisease` | `/root/autodl-tmp/bridgeDisease` |
| 原始数据 | `D:\dataset\dacl10k` | `/root/autodl-tmp/datasets/dacl10k` |
| 训练输出 | `D:\bridgeDisease\outputs` | `/root/autodl-tmp/bridgeDisease/outputs` |

AutoDL 目标目录是本项目建议值，应结合实例实际磁盘空间确认。大文件优先放数据盘；重要产物另行备份。平台目录说明见 [AutoDL 文件系统文档](https://www.autodl.com/docs/env)。

GitHub 仓库为 [ykkxxx/bridgeDisease](https://github.com/ykkxxx/bridgeDisease)，本地 `origin` 已配置为该地址。尚未连接 AutoDL，也未上传数据。下文的 AutoDL 主机和端口占位符必须替换。

## 2. 首次把代码推送到 GitHub

目标仓库为 `https://github.com/ykkxxx/bridgeDisease.git`。当前本地仓库已经配置 `origin`，可直接检查并推送：

在本地 PowerShell 中运行：

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
git remote -v
git push -u origin main
```

若要给另一份尚未配置远程的本地仓库添加同一地址，执行 `git remote add origin https://github.com/ykkxxx/bridgeDisease.git`。已有 `origin` 时先检查，确需改地址再使用 `git remote set-url origin https://github.com/ykkxxx/bridgeDisease.git`。

GitHub 认证依据所选 HTTPS 或 SSH 方式配置。私有仓库在 AutoDL 上拉取也需要相应权限，不要在 URL 中嵌入 Token。参考 [GitHub 官方发布说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 3. AutoDL 首次获取代码

在 AutoDL 的终端执行以下 Bash 命令：

```bash
mkdir -p /root/autodl-tmp/datasets
git clone https://github.com/ykkxxx/bridgeDisease.git /root/autodl-tmp/bridgeDisease
cd /root/autodl-tmp/bridgeDisease
git status
git log -1 --oneline
```

仓库目录已经存在时，进入目录检查后按下面的日常更新步骤操作，不要反复克隆到同一路径。

## 4. 日常同步代码

先在本地提交并推送：

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
git status
git add .
git diff --cached --stat
git commit -m "Describe the change"
git push
```

然后在 AutoDL 终端更新：

```bash
cd /root/autodl-tmp/bridgeDisease
git status
git pull --ff-only
git log -1 --oneline
```

建议以本地作为主要代码编辑位置。如果在 AutoDL 修改了代码，先检查并在 AutoDL 提交、推送，再回本地执行 `git pull --ff-only`。不要让两端同时修改同一文件后直接覆盖；遇到未提交修改或分支分叉时，先检查差异并处理。

每次实验记录 `git rev-parse HEAD` 的提交号，便于将结果对应到代码版本。不要在训练进行中切换正在使用的代码版本。

## 5. 首次上传原始数据

从 AutoDL 控制台复制实例的 SSH 登录命令，例如 `ssh -p 实际端口 root@实际主机`。密码仅在 SSH 提示时输入。

下面的命令在**本地 PowerShell** 执行。先把主机和端口替换为实例真实值：

```powershell
$autoDlServer = 'YOUR_AUTODL_HOST'
$autoDlPort = 12345 # 替换成控制台显示的 SSH 端口

ssh -p $autoDlPort "root@$autoDlServer" 'mkdir -p /root/autodl-tmp/datasets'
scp -P $autoDlPort -r 'D:/dataset/dacl10k' "root@${autoDlServer}:/root/autodl-tmp/datasets/"
```

结果应为 `/root/autodl-tmp/datasets/dacl10k`。`ssh` 的端口参数是小写 `-p`，`scp` 是大写 `-P`。首次 SSH 连接出现主机指纹提示时，应核对连接对象后接受。

这里使用完整数据目录，避免遗漏额外标注、清理报告或划分文件。传输前检查实例空间；本次配置没有修改任何原始数据。

上传后可对关键标注文件做哈希核对。在本地运行：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\dataset\dacl10k\samples.json'
```

在 AutoDL 终端运行：

```bash
sha256sum /root/autodl-tmp/datasets/dacl10k/samples.json
```

核对哈希一致可确认该标注文件传输完整；这不替代整份数据集的文件清单校验。

## 6. 下载 AutoDL 训练结果

在本地 PowerShell 运行，沿用上面的主机与端口变量：

```powershell
$backupDir = 'D:/bridgeDisease/outputs/autodl-backups'
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
scp -P $autoDlPort -r "root@${autoDlServer}:/root/autodl-tmp/bridgeDisease/outputs" "$backupDir/"
```

结果会放在 `D:\bridgeDisease\outputs\autodl-backups\outputs`，不会进入 Git 暂存区。需要保留多次实验时，先在服务器输出目录中按实验 ID 分子目录。

上述 SCP 命令遵循 [AutoDL 官方上传与下载文档](https://www.autodl.com/docs/scp/)。SCP 每次按指定路径拷贝，不提供自动双向同步或通用的断点续传；后续数据经常变化时，可另配支持增量传输的工具。

## 7. 提交前检查

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
git status --short
git diff --cached --stat
git ls-files
```

确认清单中只有准备共享的代码、配置和文档。`.env`、本机配置、私钥、数据目录、模型权重和训练日志已有忽略规则；不要通过 `git add -f` 绕过它们。

数据清洗与划分脚本运行方法见[数据准备说明](DATA_PREPARATION.md)。模型与训练入口已加入，默认40轮，依赖及启动步骤见[训练说明](TRAINING.md)；清洗后的自包含数据包见[7z打包说明](PACKAGE.md)。完整GPU训练仍需在AutoDL验证。
