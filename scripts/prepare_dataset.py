"""Clean a FiftyOne dacl10k export and build reproducible research manifests.

Images and source annotations are read-only. Python 3.10+ and Pillow are required.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = (
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket",
    "Hollowareas", "Cavity", "Spalling", "Graffiti", "Weathering", "Restformwork",
    "ExposedRebars", "Bearing", "EJoint", "Drainage", "PEquipment", "JTape", "WConccor",
)


class RejectedSample(ValueError):
    """An individual sample is unusable; record the reason and continue."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(data_root: Path, value: object) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RejectedSample("unsafe_image_path")
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(value).drive or ".." in relative.parts:
        raise RejectedSample("unsafe_image_path")
    resolved = (data_root / str(relative)).resolve()
    if not resolved.is_relative_to(data_root):
        raise RejectedSample("unsafe_image_path")
    if not resolved.is_file():
        raise RejectedSample("missing_image")
    return resolved, relative.as_posix()


def polygon_info(annotation: object) -> tuple[list[str], list[dict]]:
    if not isinstance(annotation, dict) or not isinstance(annotation.get("polylines"), list):
        raise RejectedSample("invalid_segmentation_annotation")
    labels, warnings = set(), []
    for index, region in enumerate(annotation["polylines"]):
        if not isinstance(region, dict):
            raise RejectedSample("invalid_polygon")
        label = region.get("label")
        if label not in CLASS_NAMES:
            raise RejectedSample("unknown_label")
        rings = region.get("points")
        if region.get("closed") is not True or region.get("filled") is not True:
            raise RejectedSample("invalid_polygon")
        if not isinstance(rings, list) or not rings:
            raise RejectedSample("invalid_polygon")
        outside = 0
        for ring in rings:
            if not isinstance(ring, list) or len(ring) < 3:
                raise RejectedSample("invalid_polygon")
            for point in ring:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise RejectedSample("invalid_polygon")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in point):
                    raise RejectedSample("invalid_polygon")
                outside += int(any(v < 0 or v > 1 for v in point))
            if len({tuple(point) for point in ring}) < 3:
                raise RejectedSample("invalid_polygon")
        labels.add(label)
        if outside:
            warnings.append({"region_index": index, "label": label,
                             "out_of_bounds_vertices": outside})
    return [name for name in CLASS_NAMES if name in labels], warnings


def inspect_sample(sample: dict, data_root: Path, empty_policy: str) -> dict:
    tags = sample.get("tags")
    if tags not in (["train"], ["validation"], ["testdev"]):
        raise RejectedSample("unknown_source_split")
    source_split = tags[0]
    path, relative = image_path(data_root, sample.get("filepath"))
    annotation = sample.get("semseg_ground_truth_filled")
    labels, coordinate_warnings = [], []
    if source_split != "testdev":
        if annotation is None:
            raise RejectedSample("missing_segmentation_annotation")
        labels, coordinate_warnings = polygon_info(annotation)
        if not annotation["polylines"] and empty_policy == "exclude":
            raise RejectedSample("empty_segmentation_regions")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise RejectedSample("unreadable_image") from error
    if width <= 0 or height <= 0:
        raise RejectedSample("unreadable_image")
    identifier = sample.get("_id")
    if isinstance(identifier, dict):
        identifier = identifier.get("$oid")
    identifier = str(identifier) if identifier is not None else relative
    metadata = sample.get("metadata") or {}
    metadata_mismatch = isinstance(metadata, dict) and any(
        metadata.get(key) is not None and metadata[key] != value
        for key, value in (("width", width), ("height", height))
    )
    return {
        "sample": sample,
        "row": {
            "image_id": identifier, "filepath": relative, "source_split": source_split,
            "annotation_available": source_split != "testdev", "width": width, "height": height,
            "image_sha256": sha256_file(path), "labels": labels,
        },
        "coordinate_warnings": coordinate_warnings, "metadata_mismatch": metadata_mismatch,
    }


def split_training(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["row"]["image_sha256"]].append(record)
    if len(groups) < 2:
        raise ValueError("At least two distinct training image groups are required")
    keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("ascii")).hexdigest())
    target = max(1, min(len(records) - 1, round(len(records) * val_ratio)))
    # Bitset subset sum: find the closest attainable image count without splitting
    # duplicate groups. Each count gets one predecessor, so reconstruction is O(n).
    reachable, parents = 1, {}
    limit = (1 << len(records)) - 1  # Exclude the all-validation assignment.
    for key in keys:
        size = len(groups[key])
        shifted = (reachable << size) & limit
        new_counts = shifted & ~reachable
        while new_counts:
            lowest = new_counts & -new_counts
            count = lowest.bit_length() - 1
            parents[count] = (count - size, key)
            new_counts ^= lowest
        reachable |= shifted
        if target in parents:
            break
    count = target if target in parents else min(parents, key=lambda value: (abs(value - target), value))
    selected = set()
    while count:
        count, key = parents[count]
        selected.add(key)
    train, val = [], []
    for record in records:
        (val if record["row"]["image_sha256"] in selected else train).append(record)
    return train, val


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(data_root: Path, output_dir: Path, annotations: Path | None = None,
                    val_ratio: float = 0.1, seed: int = 42, empty_policy: str = "exclude") -> dict:
    data_root, output_dir = Path(data_root).resolve(), Path(output_dir).resolve()
    source = Path(annotations).resolve() if annotations else data_root / "samples.json"
    if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be finite and strictly between 0 and 1")
    if empty_policy not in ("exclude", "keep"):
        raise ValueError("empty_policy must be exclude or keep")
    if output_dir.is_relative_to(data_root):
        raise ValueError("Output must be outside the original dataset directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    source_hash = sha256_file(source)
    with source.open(encoding="utf-8") as stream:
        export = json.load(stream)
    if not isinstance(export, dict) or not isinstance(export.get("samples"), list):
        raise ValueError("Expected a FiftyOne export with a samples list")
    samples = export["samples"]
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("Every samples entry must be an object")

    valid, excluded, coordinates, metadata_mismatches = [], [], [], []
    seen_paths, seen_ids = set(), set()
    raw_counts = collections.Counter()

    def reject(sample, reason):
        tags = sample.get("tags")
        excluded.append({"filepath": sample.get("filepath", ""),
                         "source_split": tags[0] if isinstance(tags, list) and tags else "unknown",
                         "reason": reason})

    def source_priority(sample):
        # Reserve valid final-test records before global path/ID deduplication.
        tags = sample.get("tags")
        return 0 if tags == ["validation"] else 1 if tags == ["train"] else 2

    ordered = sorted(samples, key=lambda sample: (source_priority(sample),
                     str(sample.get("filepath", "")), str(sample.get("_id", ""))))
    for index, sample in enumerate(ordered, 1):
        tags = sample.get("tags")
        raw_counts[tags[0] if isinstance(tags, list) and tags and isinstance(tags[0], str) else "unknown"] += 1
        try:
            record = inspect_sample(sample, data_root, empty_policy)
            row = record["row"]
            if row["filepath"] in seen_paths:
                raise RejectedSample("duplicate_filepath")
            if row["image_id"] in seen_ids:
                raise RejectedSample("duplicate_image_id")
        except RejectedSample as error:
            reject(sample, str(error))
            continue
        seen_paths.add(row["filepath"])
        seen_ids.add(row["image_id"])
        valid.append(record)
        coordinates.extend({"filepath": row["filepath"], "source_split": row["source_split"], **warning}
                           for warning in record["coordinate_warnings"])
        if record["metadata_mismatch"]:
            metadata_mismatches.append({"filepath": row["filepath"], "width": row["width"], "height": row["height"]})
        if index % 500 == 0:
            print(f"Inspected {index}/{len(samples)} records", file=sys.stderr, flush=True)

    test = [record for record in valid if record["row"]["source_split"] == "validation"]
    inference = [record for record in valid if record["row"]["source_split"] == "testdev"]
    test_hashes = {record["row"]["image_sha256"] for record in test}
    final_train = []
    for record in valid:
        if record["row"]["source_split"] == "train":
            if record["row"]["image_sha256"] in test_hashes:
                reject(record["sample"], "duplicates_final_test")
            else:
                final_train.append(record)
    if not test:
        raise ValueError("No labeled official validation images remain for final evaluation")
    train, val = split_training(final_train, val_ratio, seed)
    roles = {"train": train, "val": val, "test": test, "inference": inference, "final_train": final_train}
    clean_records = sorted(final_train + test, key=lambda record: record["row"]["filepath"])
    annotation_indices = {record["row"]["image_id"]: index for index, record in enumerate(clean_records)}
    manifests = {}
    for role, records in roles.items():
        manifests[role] = [
            {**record["row"], "split": role,
             "annotation_file": "samples_clean.json" if role != "inference" else None,
             "annotation_index": annotation_indices.get(record["row"]["image_id"]) if role != "inference" else None}
            for record in records
        ]
    for alias, role in (("train_inner", "train"), ("tune_inner", "val"), ("final_eval", "test")):
        manifests[alias] = manifests[role]
    if sha256_file(source) != source_hash:
        raise RuntimeError("Source annotations changed during preparation; no output written")

    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("manifests", "splits", "reports"):
        (output_dir / name).mkdir()
    clean_path = output_dir / "samples_clean.json"
    with clean_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write('{"samples":[\n')
        for index, record in enumerate(clean_records):
            if index:
                stream.write(",\n")
            stream.write(json.dumps(record["sample"], ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        stream.write("\n]}\n")
    for name, rows in manifests.items():
        with (output_dir / "manifests" / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        (output_dir / "splits" / f"{name}.txt").write_text(
            "".join(row["filepath"] + "\n" for row in rows), encoding="utf-8"
        )
    write_csv(output_dir / "reports" / "excluded_samples.csv", excluded, ["filepath", "source_split", "reason"])
    write_csv(output_dir / "reports" / "coordinate_warnings.csv", coordinates,
              ["filepath", "source_split", "region_index", "label", "out_of_bounds_vertices"])
    write_csv(output_dir / "reports" / "metadata_mismatches.csv", metadata_mismatches, ["filepath", "width", "height"])
    duplicate_groups = collections.defaultdict(list)
    for record in valid:
        duplicate_groups[record["row"]["image_sha256"]].append(record["row"])
    duplicate_rows = [
        {"image_sha256": digest, "filepath": row["filepath"], "source_split": row["source_split"]}
        for digest, rows in sorted(duplicate_groups.items()) if len(rows) > 1 for row in rows
    ]
    write_csv(output_dir / "reports" / "duplicate_groups.csv", duplicate_rows,
              ["image_sha256", "filepath", "source_split"])
    distributions = {
        role: {label: sum(label in row["labels"] for row in manifests[role]) for label in CLASS_NAMES}
        for role in roles
    }
    write_csv(output_dir / "reports" / "class_distribution.csv",
              [{"split": role, "label": label, "image_count": count}
               for role, counts in distributions.items() for label, count in counts.items()],
              ["split", "label", "image_count"])
    preview = {role: rows[:1] for role, rows in manifests.items() if role in roles}
    write_json(output_dir / "preview.json", preview)
    hashes = {path.relative_to(output_dir).as_posix(): sha256_file(path)
              for path in sorted(output_dir.rglob("*")) if path.is_file()}
    summary = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "data_root": str(data_root), "source_sha256": source_hash,
        "source_unchanged": sha256_file(source) == source_hash,
        "seed": seed, "requested_val_ratio": val_ratio, "empty_policy": empty_policy,
        "raw_counts": dict(raw_counts), "counts": {role: len(rows) for role, rows in roles.items()},
        "actual_inner_val_ratio": len(val) / len(final_train), "class_order": list(CLASS_NAMES),
        "excluded_count": len(excluded), "excluded_reasons": dict(collections.Counter(row["reason"] for row in excluded)),
        "clean_annotation_count": len(clean_records), "class_image_counts": distributions,
        "exact_duplicate_groups": sum(len(rows) > 1 for rows in duplicate_groups.values()),
        "metadata_mismatch_count": len(metadata_mismatches),
        "coordinate_audit": {
            "out_of_bounds_vertices": sum(row["out_of_bounds_vertices"] for row in coordinates),
            "affected_regions": len(coordinates), "affected_images": len({row["filepath"] for row in coordinates}),
            "action": "report_only_preserve_original_coordinates",
        },
        "evaluation_scope": "nonempty_only_official_validation_subset" if empty_policy == "exclude" else
                            "official_validation_after_integrity_checks_including_explicit_empty_annotations",
        "split_method": "seeded_sha256_group_order_within_official_train",
        "limitations": ["No bridge IDs: bridge-level independence is not verified.",
                        "Only byte-identical duplicates are grouped; near duplicates require manual review.",
                        "No visual label review, mask rasterization, or multilabel stratification was performed."],
        "images_modified": 0, "record_fields_changed": False, "masks_generated": False,
        "artifact_sha256": hashes,
    }
    if not summary["source_unchanged"]:
        raise RuntimeError("Source annotations changed; discard this output and investigate")
    write_json(output_dir / "reports" / "summary.json", summary)
    counts = summary["counts"]
    (output_dir / "README.md").write_text(
        "# 本地数据准备结果\n\n"
        f"- train：{counts['train']}，内部训练，用于梯度更新。\n"
        f"- val：{counts['val']}，内部验证，用于调参和选择模型。\n"
        f"- test：{counts['test']}，来自官方 validation，只用于最终评估。\n"
        f"- final_train：{counts['final_train']}，train 与 val 并集，锁定协议后重训。\n"
        f"- inference：{counts['inference']}，来自无本地真值的 testdev，只用于推理。\n\n"
        f"空标注策略：{empty_policy}；内部验证比例：{summary['actual_inner_val_ratio']:.4%}；seed={seed}。\n\n"
        "默认非空子集的结果不能标成完整官方975张验证集结果。\n"
        "JSONL 的 filepath 相对原始数据根目录；annotation_index 指向本目录 samples_clean.json 的 samples 数组。\n"
        "train_inner/tune_inner/final_eval 分别是 train/val/test 的同义清单，不要合并重复使用。\n"
        "不会生成图片副本或 mask；源图片与标注未修改。preview.json 是可直接查看的小型样例。\n"
        "完整清洗与排除原因见 reports/summary.json 及 CSV；近重复和桥梁级独立性尚未验证。\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Original dataset root containing data/ and samples.json")
    parser.add_argument("--annotations", type=Path, help="Optional export path; default: DATA_ROOT/samples.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "prepared")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction held out from official train; default: 0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--empty-policy", choices=("exclude", "keep"), default="exclude")
    args = parser.parse_args()
    try:
        summary = prepare_dataset(args.data_root, args.output, args.annotations, args.val_ratio, args.seed, args.empty_policy)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "counts": summary["counts"],
                      "excluded": summary["excluded_count"], "source_unchanged": summary["source_unchanged"]},
                     ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
