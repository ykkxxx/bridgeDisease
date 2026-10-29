# 清洗后数据打包与 AutoDL 解压

可上传文件为 `outputs/dacl10k_clean_autodl.7z`。代码继续通过 Git 同步；数据压缩包与校验文件通过 AutoDL 文件上传、SCP 或其他文件传输工具上传。压缩包、清单、日志和测试临时文件均在被 Git 忽略的 `outputs/` 下。

2026-10-29 已生成的本次压缩包大小为 **5,041,017,126 字节，约 4.695 GiB**，SHA256 为：

```text
a6803d4726a486bedeb654484b3737accf8b848f167214d275da8a72846704d9
```

本次完整 `7z t` 通过，成员清单核对为 8,851 张 JPG、26 个准备数据文件、8,881 个总文件、97 个目录。重新生成压缩包后，创建时间等信息会使整体哈希改变，应以新生成的 `.7z.sha256` 和 `.7z.verification.json` 为准。

## 包含内容

```text
bridge_disease_data/
├── dataset/
│   ├── data/data_0/…jpg … data/data_89/…jpg
│   ├── metadata.json
│   └── SOURCE_README.md
├── prepared/
│   ├── samples_clean.json
│   ├── manifests/…jsonl
│   ├── splits/…txt
│   ├── reports/…json、…csv
│   ├── README.md
│   └── preview.json
├── PACKAGE_README.md
└── package_manifest.json
```

压缩包收录 `final_train`、`test`、`inference` 三个清单引用图片的并集，共 **8,851 个唯一图片路径**。按清单相对路径去重，目录结构和图像字节保持原样；字节相同但路径不同的样本仍保留，保证清单引用完整。`train`、`val` 及其别名的引用也全部包含在该并集中。

| 清单 | 图片数 | 用途 |
| --- | ---: | --- |
| train / train_inner | 6,186 | 内部训练 |
| val / tune_inner | 687 | 调参和模型选择 |
| final_train | 6,873 | 锁定协议后，用 train 与 val 并集重训 |
| test / final_eval | 966 | 官方 validation 的非空标注子集，最终测试 |
| inference | 1,012 | 官方 testdev，无本地真值，仅用于推理 |

`test` 的 966 张结果不能标成完整官方 975 张验证集结果。`inference` 没有本地分割真值，不能计算本地真值指标。清洗排除的 71 张图片、原始 `samples.json`、缓存、权重和训练日志不会进入压缩包。`SOURCE_README.md` 和 `metadata.json` 保留来源、引用及原始许可说明；该包供训练代码读取，不是完整 FiftyOne 导出。

多标签分割应为各类别独立生成二值 mask，保留类别重叠。清洗保留原始多边形坐标及越界审计记录，光栅化时应按画布边界做几何裁剪。详细清洗策略和限制见 `prepared/reports/`。

## 在本机重新打包

依赖 Python 3.10+ 与原生 7-Zip，脚本使用 Python 标准库，无需 `py7zr`、训练依赖或网络。PowerShell 7 示例：

```powershell
Set-Location -LiteralPath 'D:\bridgeDisease'
python scripts/package_dataset.py `
  --data-root 'D:\dataset\dacl10k' `
  --prepared-root 'D:\bridgeDisease\data\prepared' `
  --output 'D:\bridgeDisease\outputs\dacl10k_clean_autodl.7z' `
  --sevenzip 'D:\7-Zip\7z.exe' `
  --compression 1 --solid off --expected-image-count 8851
```

脚本拒绝覆盖已有压缩包或同名校验副本。需要重复打包时使用不同 `--output` 文件名，已有结果可保留用于比较。`--compression` 可设 0–9，默认 1；`--solid` 为 `on` 或 `off`，默认 `off`。没有任意透传 7z 参数的接口，所有参数通过独立进程参数传递。

脚本先核对清洗产物的既有 SHA256，再逐张重算全部源图片 SHA256 并与清单比较。临时 staging 仅建在输出目录内，同盘优先为图片建硬链接，失败时复制并验证；源目录始终只读，脚本不修改源图片、源标注或准备数据。结束时只清理自身生成的 staging。压缩先生成临时文件，经过完整 `7z t` 和全部成员路径、文件大小比对后才以拒绝覆盖的方式发布最终压缩包。

同目录生成：

| 文件后缀 | 内容 |
| --- | --- |
| `.7z.sha256` | 整个压缩包的 SHA256，供传输后核对 |
| `.7z.verification.json` | 压缩包大小、哈希、图片和文件数量、完整性结果 |
| `.7z.listing.txt` | 原生 7z 输出的全部目录与文件成员清单 |
| `.7z.log`、`.7z.test.log` | 压缩和完整性测试日志 |

包内 `package_manifest.json` 为每个其他文件记录相对路径、字节数和 SHA256。该文件不记录自身哈希，以避免循环依赖；整个压缩包的外部 SHA256 保护它本身。

## 上传到 AutoDL 并校验

将 `.7z` 和 `.7z.sha256` 两个文件放在 AutoDL 的同一目录。下面假定上传到 `/root/autodl-tmp/`，并且已安装提供 `7z` 命令的原生 7-Zip：

```bash
cd /root/autodl-tmp
sha256sum -c dacl10k_clean_autodl.7z.sha256
7z t dacl10k_clean_autodl.7z
7z x dacl10k_clean_autodl.7z -o/root/autodl-tmp
```

配置中的目录使用：

```yaml
data_root: /root/autodl-tmp/bridge_disease_data/dataset
prepared_root: /root/autodl-tmp/bridge_disease_data/prepared
```

JSONL 的 `filepath` 相对 `data_root`，`annotation_index` 指向 `prepared_root/samples_clean.json` 内 `samples` 数组。清洗报告中历史 Windows 绝对路径用于追溯来源，不是云端运行目录；不要据此改写 JSONL 图片路径。

需要检查解压后每个文件时，可在 AutoDL 运行下列标准库脚本。它会检查所有清单记录的大小和 SHA256，默认读取完整文件内容，约需再次顺序读取 5 GB 数据：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('/root/autodl-tmp/bridge_disease_data')
manifest = json.loads((root / 'package_manifest.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    path = root / entry['path']
    if path.stat().st_size != entry['size']:
        raise SystemExit(f"Size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != entry['sha256']:
        raise SystemExit(f"SHA256 mismatch: {path}")
print(f"Verified {len(manifest['files'])} files; {manifest['unique_image_count']} unique images")
PY
```

## 小型集成测试

```powershell
python -m unittest discover -s tests -p test_package_dataset.py -v
```

测试用真实 JPEG 和本机 7z 完成压缩、解压、逐文件哈希比对，检查推理图片纳入、清单去重、源文件不变、拒绝覆盖、路径越界、内容损坏和清单引用完整性。找不到原生 7z 时归档集成测试会显式跳过。
