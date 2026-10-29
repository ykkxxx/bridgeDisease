"""Dataset-level confusion counts for overlapping multilabel masks."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .constants import CLASS_NAMES


class MultiLabelMetrics:
    """Accumulate TP/FP/FN before computing class IoU, Dice, and macro IoU.

    ``update`` takes raw logits and applies independent sigmoid thresholds.
    Undefined ratios are ``None`` (JSON null). Macro averages exclude undefined
    class IoUs and expose the number of classes actually included.
    """

    def __init__(self, num_classes=19, threshold=.5):
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be strictly between0and1")
        self.num_classes = num_classes
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, dtype=torch.int64)
        self.fp = torch.zeros(self.num_classes, dtype=torch.int64)
        self.fn = torch.zeros(self.num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: Tensor, target: Tensor, valid: Tensor):
        if logits.ndim != 4 or logits.shape != target.shape or logits.shape[1] != self.num_classes:
            raise ValueError("logits and target must have identical [B,num_classes,H,W] shapes")
        if valid.ndim != 4 or valid.shape[0] != logits.shape[0] or \
                valid.shape[-2:] != logits.shape[-2:] or valid.shape[1] not in (1, self.num_classes):
            raise ValueError("valid must be [B,1,H,W] or match the logits shape")
        prediction = logits.float().sigmoid() >= self.threshold
        truth = target.to(device=logits.device) > .5
        observed = valid.to(device=logits.device, dtype=torch.bool).expand_as(prediction)
        dimensions = (0, 2, 3)
        self.tp += (prediction & truth & observed).sum(dim=dimensions, dtype=torch.int64).cpu()
        self.fp += (prediction & ~truth & observed).sum(dim=dimensions, dtype=torch.int64).cpu()
        self.fn += (~prediction & truth & observed).sum(dim=dimensions, dtype=torch.int64).cpu()

    def compute(self) -> dict:
        rows = []
        for index, (tp, fp, fn) in enumerate(zip(self.tp.tolist(), self.fp.tolist(), self.fn.tolist())):
            union, dice_denominator = tp + fp + fn, 2 * tp + fp + fn
            rows.append({
                "label": CLASS_NAMES[index] if index < len(CLASS_NAMES) else f"class_{index}",
                "iou": tp / union if union else None,
                "dice": 2 * tp / dice_denominator if dice_denominator else None,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "tp": tp, "fp": fp, "fn": fn,
            })
        disease = [row["iou"] for row in rows[:13] if row["iou"] is not None]
        all_classes = [row["iou"] for row in rows if row["iou"] is not None]
        return {"miou13": sum(disease) / len(disease) if disease else None,
                "miou19": sum(all_classes) / len(all_classes) if all_classes else None,
                "miou13_defined_classes": len(disease), "miou19_defined_classes": len(all_classes),
                "threshold": self.threshold, "per_class": rows}
