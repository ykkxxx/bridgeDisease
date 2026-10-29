# Dataset Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成可复现的数据清洗、训练/验证/测试清单及作用说明。

**Architecture:** 单个 Python CLI 读取 FiftyOne 原始导出，检查图片及分割多边形，按图像 SHA-256 分组后沿用已选研究划分，输出相对路径清单与审计报告。原始数据只读，不复制图片，不生成 mask。

**Tech Stack:** Python 3.10+、Pillow、unittest。

**Spec:** 用户已选择沿用 `2026-10-29-30-day-experiments.md` 的内部约90/10与独立最终评估协议；本次默认延续既有非空清洗子集，说明966与官方975的区别。

## Global Constraints

- 使用原始 `samples.json`；不覆盖原文件、旧 `samples_clean.json` 或旧清洗报告。
- 默认排除有标注集合中的空多边形记录；`--empty-policy keep` 可显式保留。
- 官方 train 内约90/10：`train` 学习参数、`val` 调参和选择 checkpoint；官方 validation 为本地 `test`，仅用于最终报告。
- `testdev` 无真值，单独输出 `inference`；不得当成全零标签或有真值测试集。
- `final_train` 等于内部 train 与 val 的并集，用于协议锁定后完整重训；不包含最终 test。
- 固定 seed=42；相同内容图片不可跨内部 train/val；与 test 内容相同的训练记录排除并审计。
- 保留原始多边形及19类顺序，越界只报告，缺失/损坏图片和非法标注整条排除并记录。
- 输出目录不得位于原始数据目录中，已有输出不得覆盖。默认输出 `data/prepared/`，全部被 Git 忽略。

## Review Focus

- null 与显式空标注区分；testdev 不进入有监督清单。
- 重复图片跨边界、重复路径/ID、缺少桥梁ID的披露。
- 非法路径、异常/非有限坐标、未知标签、损坏图片。
- 输出覆盖或源文件变更，Windows/Linux 路径与 Python 3.10 兼容。
- 默认非空子集与保留空标注模式的评估范围准确。

### Task 1: Cleaning, grouping, and export

**Files:** `scripts/prepare_dataset.py`、`tests/test_prepare_dataset.py`、`requirements-data.txt`。

**Interfaces:** `prepare_dataset(data_root, output_dir, annotations=None, val_ratio=0.1, seed=42, empty_policy="exclude") -> dict`；CLI 参数同名并使用连字符。

- [x] 写真实临时图片与标注的测试，覆盖上述 Review Focus。
- [x] 运行 `python -m unittest discover -s tests -v`，确认因 CLI 尚不存在而失败。
- [x] 实现清洗、相同内容分组、确定性内部划分、导出清单及摘要。
- [x] 运行全部测试，确认通过。独立审查的两项问题也已分别复现并修复，最终17项通过。

### Task 2: Real-data verification and usage documentation

**Files:** `docs/DATA_PREPARATION.md`、`README.md`、`data/README.md`；本地输出 `data/prepared/`。

**Interfaces:** JSONL 每行包括 `image_id`、`filepath`、`source_split`、`split`、`annotation_available`、`width`、`height`、`image_sha256`、`annotation_index`；相对路径以原始 data root 为基准。

- [x] 全量运行 CLI，并独立核对样本数、源哈希、集合交集、标注索引和结果文件哈希。
- [x] 写明三类集合用途、最终重训流程、非空子集评估限制和 Windows/AutoDL 命令。
- [x] 验证 CLI 帮助、全部测试、Markdown 本地链接与 Git 忽略规则；独立审查代码后交付。
