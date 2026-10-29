"""Shared optimization and original-image sliding-window evaluation."""
from __future__ import annotations

import math
import time

import torch
from torch.nn import functional as F

from .losses import segmentation_loss
from .metrics import MultiLabelMetrics


def make_scaler(device, amp=True):
    enabled = bool(amp and device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, optimizer, device, *, accumulation_steps=8, amp=True,
                    coarse_weight=.2, edge_weight=.1, grad_clip=1., max_batches=None,
                    scaler=None, lr_callback=None, log_every=100):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    if total_batches < 1:
        raise ValueError("Training loader is empty")
    model.train()
    use_amp = bool(amp and device.type == "cuda")
    scaler = scaler if scaler is not None else make_scaler(device, amp)
    optimizer.zero_grad(set_to_none=True)
    sums, steps, started = {}, 0, time.perf_counter()
    for index, batch in enumerate(loader):
        if index >= total_batches:
            break
        if lr_callback:
            lr_callback(index / total_batches)
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(image)
            loss, components = segmentation_loss(outputs, target, valid,
                                                  coarse_weight=coarse_weight, edge_weight=edge_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {index}")
        group_start = (index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)
        scaler.scale(loss / group_size).backward()
        if (index + 1) % accumulation_steps == 0 or index + 1 == total_batches:
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None], grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        sums["loss"] = sums.get("loss", 0.) + float(loss.detach())
        for name, value in components.items():
            sums[name] = sums.get(name, 0.) + float(value.detach() if torch.is_tensor(value) else value)
        if log_every and (index + 1) % log_every == 0:
            print(f"  batch {index + 1}/{total_batches}, loss={sums['loss'] / (index + 1):.5f}", flush=True)
    result = {name: value / total_batches for name, value in sums.items()}
    result.update(optimizer_steps=steps, batches=total_batches, seconds=time.perf_counter() - started)
    return result


def _starts(length, window, stride):
    if length <= window:
        return [0]
    values = list(range(0, length - window + 1, stride))
    if values[-1] != length - window:
        values.append(length - window)
    return values


@torch.no_grad()
def sliding_window_predict(model, image, device, *, image_size=512, stride=352,
                           max_long_side=1536, amp=True):
    """Return CPU logits [1,C,H,W] at the ORIGINAL image resolution."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Full-image evaluation requires batch_size=1")
        image = image[0]
    if image.ndim != 3 or not 0 < stride <= image_size:
        raise ValueError("Expected [3,H,W] image and 0 < stride <= image_size")
    model.eval()
    original_size = image.shape[-2:]
    image = image.detach().float().cpu().unsqueeze(0)
    if max_long_side and max(original_size) > max_long_side:
        scale = max_long_side / max(original_size)
        resized = tuple(max(1, round(size * scale)) for size in original_size)
        image = F.interpolate(image, resized, mode="bilinear", align_corners=False)
    height, width = image.shape[-2:]
    canvas, count = None, torch.zeros(1, 1, height, width)
    for top in _starts(height, image_size, stride):
        for left in _starts(width, image_size, stride):
            tile = image[:, :, top:top + image_size, left:left + image_size]
            tile_h, tile_w = tile.shape[-2:]
            tile = F.pad(tile, (0, image_size - tile_w, 0, image_size - tile_h))
            with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                logits = model(tile.to(device))["logits"]
            if logits.shape[-2:] != (image_size, image_size):
                logits = F.interpolate(logits, (image_size, image_size), mode="bilinear", align_corners=False)
            logits = logits[:, :, :tile_h, :tile_w].float().cpu()
            if canvas is None:
                canvas = torch.zeros(1, logits.shape[1], height, width)
            canvas[:, :, top:top + tile_h, left:left + tile_w] += logits
            count[:, :, top:top + tile_h, left:left + tile_w] += 1
    if canvas is None or (count == 0).any():
        raise RuntimeError("Sliding windows did not cover the complete image")
    logits = canvas / count
    if logits.shape[-2:] != original_size:
        logits = F.interpolate(logits, original_size, mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def evaluate_model(model, loader, device, *, threshold=.5, image_size=512, stride=352,
                   max_long_side=1536, amp=True, max_images=None):
    metrics = MultiLabelMetrics(threshold=threshold)
    started, processed = time.perf_counter(), 0
    for batch in loader:
        if max_images and processed >= max_images:
            break
        logits = sliding_window_predict(model, batch["image"], device, image_size=image_size,
                                        stride=stride, max_long_side=max_long_side, amp=amp)
        metrics.update(logits, batch["target"].cpu(), batch["valid"].cpu())
        processed += 1
        if processed % 50 == 0:
            print(f"  evaluated {processed}/{len(loader)} images", flush=True)
    result = metrics.compute()
    result.update(images=processed, seconds=time.perf_counter() - started,
                  partial=processed < len(loader), threshold=threshold,
                  image_size=image_size, stride=stride, max_long_side=max_long_side)
    return result


def set_learning_rates(optimizer, epoch_progress, total_epochs, warmup_epochs=5,
                       min_lr_ratio=.01, unfreeze_epoch=6):
    for group in optimizer.param_groups:
        if group.get("name") == "backbone":
            offset = unfreeze_epoch - 1
            if epoch_progress < offset:
                factor = 0.
            else:
                local = epoch_progress - offset
                warm = min(2., max(1., total_epochs - offset))
                cosine = .5 * (1 + math.cos(math.pi * min(1., local / max(1., total_epochs - offset))))
                factor = min(1., (local + 1) / warm) * (min_lr_ratio + (1 - min_lr_ratio) * cosine)
        elif epoch_progress < warmup_epochs:
            factor = min(1., (epoch_progress + 1) / max(1, warmup_epochs))
        else:
            progress = min(1., (epoch_progress - warmup_epochs) / max(1, total_epochs - warmup_epochs))
            factor = min_lr_ratio + (1 - min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))
        group["lr"] = group["base_lr"] * factor
