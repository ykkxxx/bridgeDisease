"""Train/evaluate/infer entry points with explicit development/final protocols."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Cyclic _base configuration")
    seen.add(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    base = config.pop("_base", None)
    result = load_config(path.parent / base, seen) if base else {}
    result.update(config)
    if result.get("epochs", 40) < 1:
        raise ValueError("epochs must be positive")
    return result


def stage_splits(stage):
    if stage == "dev":
        return "train", "val"
    if stage == "final":
        return "final_train", None
    raise ValueError("stage must be dev or final")


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def _seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_seed(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _model(config):
    os.environ.setdefault("HF_HOME", str(ROOT / "outputs" / "huggingface-cache"))
    from .models import build_model
    prompt_file = Path(config["prompt_file"])
    if not prompt_file.is_absolute():
        prompt_file = ROOT / prompt_file
    return build_model(config["model"], prompt_file,
                       clip_model_name=config.get("clip_model_name", "openai/clip-vit-base-patch16"),
                       segformer_model_name=config.get("segformer_model_name", "nvidia/mit-b2"),
                       pretrained=config.get("pretrained", True),
                       gradient_checkpointing=config.get("gradient_checkpointing", True))


def _checkpoint_state(model, optimizer, scaler, epoch, config, best, best_epoch, generator):
    import numpy as np
    import torch
    backbone = list(model.backbone_parameters()) if hasattr(model, "backbone_parameters") else []
    saved_ids = {id(parameter) for parameter in backbone}
    names = {name for name, parameter in model.named_parameters()
             if parameter.requires_grad or id(parameter) in saved_ids}
    names.update(name for name, _ in model.named_buffers())
    full_state = not config.get("pretrained", True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items() if full_state or name in names}
    return {"format_version": 1, "model": state,
            "model_state_kind": "full_state" if full_state else "trainable_parameters_and_buffers",
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
            "config": config, "best_metric": best, "best_epoch": best_epoch,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "loader": generator.get_state()}}


def _atomic_torch_save(path, state):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _load_weights(model, checkpoint):
    full_state = checkpoint.get("model_state_kind") == "full_state"
    result = model.load_state_dict(checkpoint["model"], strict=full_state)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint parameters: {result.unexpected_keys}")
    candidates = {id(parameter) for parameter in model.backbone_parameters()} if hasattr(model, "backbone_parameters") else set()
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad or id(parameter) in candidates}
    required.update(name for name, _ in model.named_buffers())
    missing = set(result.missing_keys) & required & set(model.state_dict())
    if missing:
        raise ValueError(f"Checkpoint misses required parameters: {sorted(missing)}")
    expected = checkpoint.get("config", {}).get("frozen_backbone_sha256")
    if expected is not None and _frozen_fingerprint(model) != expected:
        raise ValueError("Frozen pretrained backbone differs from the checkpoint; use the original weights/cache")


def _frozen_fingerprint(model):
    import torch
    digest = hashlib.sha256()
    candidates = {id(parameter) for parameter in model.backbone_parameters()} if hasattr(model, "backbone_parameters") else set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and id(parameter) not in candidates:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train multilabel bridge segmentation; default40epochs")
    parser.add_argument("--config", default="configs/experiments/P.json")
    parser.add_argument("--stage", choices=("dev", "final"), default="dev")
    parser.add_argument("--data-root")
    parser.add_argument("--prepared-root")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", help="Trusted local last.pt checkpoint")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    for name in ("epochs", "seed", "image-size", "num-workers", "val-every", "early-stop-patience"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--max-train-batches", type=int, help="Smoke check only; records partial run")
    parser.add_argument("--max-val-images", type=int, help="Smoke check only; records partial evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without downloading weights")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for key in ("data_root", "prepared_root", "epochs", "seed", "image_size", "num_workers", "val_every", "early_stop_patience"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config["stage"] = args.stage
    config["max_train_batches"] = args.max_train_batches
    config["max_val_images"] = args.max_val_images
    config["partial_run"] = bool(args.max_train_batches or args.max_val_images)
    if config["epochs"] < 1 or config["image_size"] < 16 or config["image_size"] % 16:
        parser.error("epochs must be positive; image-size must be a positive multiple of16")
    if config["num_workers"] < 0 or config["val_every"] < 1 or config["early_stop_patience"] < 0:
        parser.error("invalid worker, validation interval, or early-stop value")
    config["stride"] = min(config["stride"], config["image_size"])
    if args.stage == "final" and config["early_stop_patience"]:
        parser.error("Final retraining uses locked epochs; early stopping is development-only")
    run_dir = Path(args.run_dir or f"outputs/{args.stage}/{config['model']}_seed{config['seed']}")
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), **config, "splits": stage_splits(args.stage)}, indent=2))
        return 0

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .data import BridgeDataset
    from .engine import evaluate_model, make_scaler, set_learning_rates, train_one_epoch
    _seed(config["seed"])
    device = _device(args.device)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        parser.error(f"Run directory is not empty: {run_dir}; choose another or use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_split, val_split = stage_splits(args.stage)
    prompt = Path(config["prompt_file"])
    prompt = prompt if prompt.is_absolute() else ROOT / prompt
    config["data_fingerprint"] = {
        "train_manifest": _hash(Path(config["prepared_root"]) / f"manifests/{train_split}.jsonl"),
        "annotations": _hash(Path(config["prepared_root"]) / "samples_clean.json"),
        "prompts": _hash(prompt),
    }
    if val_split:
        config["data_fingerprint"]["validation_manifest"] = _hash(Path(config["prepared_root"]) / f"manifests/{val_split}.jsonl")
    generator = torch.Generator().manual_seed(config["seed"])
    training = BridgeDataset(config["data_root"], config["prepared_root"], train_split,
                             image_size=config["image_size"], training=True, max_long_side=config["max_long_side"])
    loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config["num_workers"], pin_memory=device.type == "cuda",
                        generator=generator, worker_init_fn=_worker_seed, persistent_workers=False)
    val_loader = None
    if val_split:
        validation = BridgeDataset(config["data_root"], config["prepared_root"], val_split,
                                   image_size=None, training=False)
        val_loader = DataLoader(validation, batch_size=1, num_workers=0)
    model = _model(config).to(device)
    config["frozen_backbone_sha256"] = _frozen_fingerprint(model)
    backbone = list(model.backbone_parameters()) if hasattr(model, "backbone_parameters") else []
    backbone_ids = {id(parameter) for parameter in backbone}
    main = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in backbone_ids]
    groups = [{"params": main, "lr": config["lr"], "base_lr": config["lr"], "name": "main"}]
    if backbone:
        groups.append({"params": backbone, "lr": 0., "base_lr": config["backbone_lr"], "name": "backbone"})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    scaler = make_scaler(device, config["amp"])
    start, best, best_epoch = 1, None, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior = checkpoint["config"]
        mutable = {"data_root", "prepared_root", "prompt_file", "num_workers"}
        for key in sorted(set(prior) | set(config)):
            if key not in mutable and prior.get(key) != config.get(key):
                raise ValueError(f"Resume config mismatch: {key}")
        _load_weights(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, best_epoch = checkpoint["epoch"] + 1, checkpoint["best_metric"], checkpoint["best_epoch"]
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        generator.set_state(rng["loader"])
        if rng["cuda"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(rng["cuda"])
    _json(run_dir / "config.json", config)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    metadata = {"git_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
                "torch": torch.__version__, "device": str(device), "train_split": train_split,
                "evaluation_split_during_training": val_split,
                "train_manifest_sha256": _hash(Path(config["prepared_root"]) / f"manifests/{train_split}.jsonl"),
                "prompt_sha256": _hash(prompt), "class_count": 19,
                "data_fingerprint": config["data_fingerprint"],
                "frozen_backbone_sha256": config["frozen_backbone_sha256"],
                "initial_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "checkpoint_note": "Frozen pretrained parameters must be available from the recorded model source/cache."}
    _json(run_dir / "metadata.json", metadata)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, config["epochs"] + 1):
        if hasattr(model, "set_train_epoch"):
            model.set_train_epoch(epoch)
        print(f"Epoch {epoch}/{config['epochs']} model={config['model']} stage={args.stage}", flush=True)
        record = {"epoch": epoch, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        record["train"] = train_one_epoch(
            model, loader, optimizer, device, accumulation_steps=config["accumulation_steps"], amp=config["amp"],
            coarse_weight=config["coarse_weight"], edge_weight=config["edge_weight"], grad_clip=config["grad_clip"],
            max_batches=args.max_train_batches, scaler=scaler,
            lr_callback=lambda fraction: set_learning_rates(optimizer, epoch - 1 + fraction, config["epochs"],
                        config["warmup_epochs"], config["min_lr_ratio"]))
        improved = False
        if val_loader is not None and (epoch % config["val_every"] == 0 or epoch == config["epochs"]):
            result = evaluate_model(model, val_loader, device, threshold=config["threshold"],
                        image_size=config["image_size"], stride=min(config["stride"], config["image_size"]),
                        max_long_side=config["max_long_side"], amp=config["amp"], max_images=args.max_val_images)
            record["val"] = result
            score = result["miou13"]
            improved = score is not None and (best is None or score > best)
            if improved:
                best, best_epoch = score, epoch
            print(f"  val mIoU13={score} best={best} epoch={best_epoch}", flush=True)
        state = _checkpoint_state(model, optimizer, scaler, epoch, config, best, best_epoch, generator)
        _atomic_torch_save(run_dir / "last.pt", state)
        if improved:
            _atomic_torch_save(run_dir / "best.pt", state)
        if args.stage == "final" and epoch == config["epochs"]:
            _atomic_torch_save(run_dir / "final.pt", state)
        if device.type == "cuda":
            record["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        patience = config["early_stop_patience"]
        if val_loader is not None and patience and best_epoch and epoch - best_epoch >= patience:
            print(f"Development early stop at epoch{epoch}; best epoch{best_epoch}", flush=True)
            break
    print(f"Saved run to {run_dir}", flush=True)
    return 0


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Score a fixed checkpoint with original-image sliding inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--prepared-root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-images", type=int, help="Smoke check only, marked partial")
    args = parser.parse_args(argv)
    import torch
    from torch.utils.data import DataLoader
    from .data import BridgeDataset
    from .engine import evaluate_model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if args.data_root:
        config["data_root"] = args.data_root
    if args.prepared_root:
        config["prepared_root"] = args.prepared_root
    device = _device(args.device)
    model = _model(config).to(device)
    _load_weights(model, checkpoint)
    dataset = BridgeDataset(config["data_root"], config["prepared_root"], args.split, image_size=None, training=False)
    metrics = evaluate_model(model, DataLoader(dataset, batch_size=1, num_workers=0), device,
                threshold=config["threshold"], image_size=config["image_size"], stride=config["stride"],
                max_long_side=config["max_long_side"], amp=config["amp"], max_images=args.max_images)
    metrics.update(split=args.split, checkpoint=str(args.checkpoint), checkpoint_sha256=_hash(args.checkpoint),
                   manifest_sha256=_hash(Path(config["prepared_root"]) / f"manifests/{args.split}.jsonl"))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, metrics)
    per_class = metrics.get("per_class", [])
    if per_class:
        with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
            writer.writeheader()
            writer.writerows(per_class)
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))
    return 0


def infer_main(argv=None):
    parser = argparse.ArgumentParser(description="Predict independent binary masks without ground-truth labels")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="One image file or a directory (recursive)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image
    from .constants import CLASS_NAMES
    from .engine import sliding_window_predict
    source = Path(args.image)
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise FileNotFoundError(f"No input images: {source}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite inference output: {output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = _device(args.device)
    model = _model(config).to(device)
    _load_weights(model, checkpoint)
    threshold = config["threshold"] if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between0and1")
    output.mkdir(parents=True)
    for path in files:
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
        logits = sliding_window_predict(model, tensor, device, image_size=config["image_size"], stride=config["stride"],
                                        max_long_side=config["max_long_side"], amp=config["amp"])
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        destination = output / relative.parent / relative.name
        destination.mkdir(parents=True)
        binary = logits.sigmoid()[0] >= threshold
        for index, label in enumerate(CLASS_NAMES):
            Image.fromarray(binary[index].numpy().astype(np.uint8) * 255).save(destination / f"{label}.png")
        _json(destination / "prediction.json", {"image": str(path), "threshold": threshold,
                    "labels_with_pixels": [label for index, label in enumerate(CLASS_NAMES) if binary[index].any()]})
        print(f"Predicted {path}", flush=True)
    return 0
