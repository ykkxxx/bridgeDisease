# 数据目录

本地现有原始数据：`D:\dataset\dacl10k`。

AutoDL 建议数据目录：`/root/autodl-tmp/datasets/dacl10k`。

本目录除了此说明外均被 Git 忽略。原始图片、大型标注 JSON、生成的 mask 和缓存通过文件传输单独管理。

数据上传方法见 [同步说明](../docs/SYNC.md)。

清洗与划分入口为 `scripts/prepare_dataset.py`，默认结果写到本目录下的 `prepared/`。代码不复制图片，不覆盖原始标注；详细规则和三类集合用途见[数据准备说明](../docs/DATA_PREPARATION.md)。

本次输出中的 `preview.json` 与 `reports/summary.json` 可以直接查看，无需打开大型标注文件。
