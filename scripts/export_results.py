"""Read existing experiment records into traceable thesis CSV tables (no Torch required)."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_history(path, warnings):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                warnings.append(f"Ignored unfinished JSONL tail: {path}")
            else:
                raise ValueError(f"Invalid completed history line {index + 1}: {path}")
    epochs = [record["epoch"] for record in records]
    if len(set(epochs)) != len(epochs) or epochs != sorted(epochs):
        raise ValueError(f"Duplicate or out-of-order epochs: {path}")
    return records


def write_csv(path, rows, default_fields):
    fields = list(default_fields)
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(runs_root, output):
    runs_root, output = Path(runs_root).resolve(), Path(output).resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"No runs directory: {runs_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    warnings, curves, classes, summaries, evaluations = [], [], [], [], []
    groups = collections.defaultdict(list)
    for config_path in sorted(runs_root.rglob("config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not {"model", "seed", "stage", "epochs"} <= config.keys():
            continue  # e.g. model download caches, not experiment records
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        history = read_history(run_dir / "history.jsonl", warnings)
        identity = {"run": run_dir.relative_to(runs_root).as_posix(), "model": config["model"],
                    "stage": config["stage"], "seed": config["seed"]}
        max_epoch = max((record["epoch"] for record in history), default=0)
        partial = bool(config.get("partial_run") or config.get("max_train_batches") or config.get("max_val_images"))
        partial = partial or any(record.get("val", {}).get("partial", False) for record in history)
        status = "partial" if partial else "complete" if max_epoch == config["epochs"] else "in_progress"
        if not partial and [record["epoch"] for record in history] != list(range(1, max_epoch + 1)):
            status = "incomplete_history"
            warnings.append(f"Epoch history has gaps or begins after epoch1: {run_dir}")
        protocol = {key: value for key, value in config.items() if key not in {
            "seed", "data_root", "prepared_root", "prompt_file", "num_workers", "frozen_backbone_sha256"}}
        protocol["git_commit"] = metadata.get("git_commit", "unknown")
        protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
        validated = []
        for record in history:
            train, val = record.get("train", {}), record.get("val", {})
            row = {**identity, "epoch": record["epoch"], "status": status,
                   "train_loss": number(train.get("loss")), "train_seconds": number(train.get("seconds")),
                   "val_miou13": number(val.get("miou13")), "val_miou19": number(val.get("miou19")),
                   "val_seconds": number(val.get("seconds")), "val_images": val.get("images"),
                   "peak_memory_gib": record.get("peak_memory_bytes", 0) / 1024 ** 3 if "peak_memory_bytes" in record else None,
                   "trainable_parameters": record.get("trainable_parameters"), "protocol_id": protocol_id}
            row.update({f"train_{name}": number(train.get(name)) for name in ("main", "bce", "dice", "coarse", "edge")})
            curves.append(row)
            if number(val.get("miou13")) is not None and not val.get("partial"):
                validated.append((record["epoch"], val))
            for values in val.get("per_class", []):
                classes.append({**identity, "source": "history", "split": "val", "epoch": record["epoch"], **values})
        best_epoch, best_val = max(validated, key=lambda pair: pair[1]["miou13"]) if validated else (None, {})
        summary = {**identity, "status": status, "epochs_completed": max_epoch, "epochs_planned": config["epochs"],
                   "best_internal_epoch": best_epoch, "best_val_miou13": number(best_val.get("miou13")),
                   "best_val_miou19": number(best_val.get("miou19")), "protocol_id": protocol_id,
                   "git_commit": metadata.get("git_commit"), "total_parameters": metadata.get("total_parameters"),
                   "initial_trainable_parameters": metadata.get("initial_trainable_parameters"),
                   "total_train_seconds": sum(number(record.get("train", {}).get("seconds")) or 0 for record in history)}
        summaries.append(summary)
        test_reports = []
        for report_path in sorted(run_dir.glob("*.json")):
            if report_path.name in ("config.json", "metadata.json"):
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict) or report.get("split") not in ("val", "test") or "per_class" not in report:
                continue
            report_id = report_path.name
            evaluation = {**identity, "report": report_id, "split": report["split"],
                          "miou13": number(report.get("miou13")), "miou19": number(report.get("miou19")),
                          "partial": bool(report.get("partial")), "images": report.get("images"),
                          "threshold": report.get("threshold"), "checkpoint_sha256": report.get("checkpoint_sha256"),
                          "manifest_sha256": report.get("manifest_sha256"), "protocol_id": protocol_id}
            evaluations.append(evaluation)
            for values in report["per_class"]:
                classes.append({**identity, "source": report_id, "split": report["split"], "epoch": None, **values})
            if report["split"] == "test" and not report.get("partial"):
                test_reports.append(report)
        if status != "complete":
            continue
        if not metadata.get("git_commit") or metadata["git_commit"] == "unknown":
            warnings.append(f"No verified code identity; excluded from seed statistics: {run_dir}")
            continue
        if config["stage"] == "dev" and best_val:
            groups[(config["model"], "dev", "best_internal_val", protocol_id)].append((config["seed"], best_val, identity["run"]))
        elif config["stage"] == "final" and len(test_reports) == 1 and (run_dir / "final.pt").exists():
            report = test_reports[0]
            # Only scores from the locked final checkpoint belong to final tables.
            if Path(str(report.get("checkpoint", "")).replace("\\", "/")).name != "final.pt":
                warnings.append(f"Final test report does not reference final.pt: {run_dir}")
                continue
            if report.get("checkpoint_sha256") != file_hash(run_dir / "final.pt"):
                warnings.append(f"Final checkpoint hash mismatch: {run_dir}")
                continue
            eval_protocol = {name: report.get(name) for name in (
                "threshold", "image_size", "stride", "max_long_side", "manifest_sha256", "images")}
            evaluation_id = hashlib.sha256(json.dumps(eval_protocol, sort_keys=True).encode()).hexdigest()[:16]
            groups[(config["model"], "final", "test", protocol_id + "-" + evaluation_id)].append((config["seed"], report, identity["run"]))
        elif config["stage"] == "final":
            warnings.append(f"No unambiguous completed final test report: {run_dir}")
    aggregates = []
    for (model, stage, split, protocol_id), entries in sorted(groups.items()):
        seeds = [entry[0] for entry in entries]
        duplicates = {seed for seed in seeds if seeds.count(seed) > 1}
        if duplicates:
            warnings.append(f"Excluded duplicate seed runs for {model}/{stage}/{protocol_id}: {sorted(duplicates)}")
        usable = [entry for entry in entries if entry[0] not in duplicates]
        for metric in ("miou13", "miou19"):
            measured = [(seed, number(report.get(metric)), run) for seed, report, run in usable
                        if number(report.get(metric)) is not None]
            values = [entry[1] for entry in measured]
            if values:
                aggregates.append({"model": model, "stage": stage, "split": split, "protocol_id": protocol_id,
                                   "metric": metric, "n_seeds": len(values), "seeds": ";".join(str(entry[0]) for entry in measured),
                                   "mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None,
                                   "runs": ";".join(entry[2] for entry in measured)})
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "curves.csv", curves, ("run", "model", "stage", "seed", "epoch", "train_loss", "val_miou13", "val_miou19"))
    write_csv(output / "per_class_metrics.csv", classes, ("run", "model", "stage", "seed", "source", "split", "epoch", "label", "iou", "dice", "precision", "recall", "tp", "fp", "fn"))
    write_csv(output / "run_summary.csv", summaries, ("run", "model", "stage", "seed", "status"))
    write_csv(output / "evaluations.csv", evaluations, ("run", "model", "stage", "seed", "report", "split", "miou13", "miou19"))
    write_csv(output / "seed_summary.csv", aggregates, ("model", "stage", "split", "protocol_id", "metric", "n_seeds", "seeds", "mean", "std", "runs"))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs_root": str(runs_root),
              "runs": len(summaries), "curve_rows": len(curves), "per_class_rows": len(classes), "warnings": warnings,
              "notes": ["Metrics remain fractions0..1; multiply by100 for percent plots.",
                        "A single seed has no sample standard deviation; no missing measurement is fabricated.",
                        "Export is a read-only snapshot; rerun to a new directory after later epochs."]}
    (output / "export_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = export_results(args.runs_root, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
