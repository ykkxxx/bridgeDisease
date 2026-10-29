"""Valid-pixel multilabel segmentation and optional auxiliary supervision."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _mask(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if logits.ndim != 4 or target.shape != logits.shape:
        raise ValueError("logits and target must have identical [B,C,H,W] shapes")
    if valid.ndim != 4 or valid.shape[0] != logits.shape[0] or \
            valid.shape[-2:] != logits.shape[-2:] or valid.shape[1] not in (1, logits.shape[1]):
        raise ValueError("valid must be [B,1,H,W] or match the logits shape")
    return valid.to(device=logits.device, dtype=torch.float32).expand_as(logits)


def masked_bce(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """BCE averaged across valid pixels and classes; all-invalid inputs give0."""
    mask = _mask(logits, target, valid)
    values = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _positive_dice(logits: Tensor, target: Tensor, valid: Tensor, eps: float = 1e-6) -> Tensor:
    mask = _mask(logits, target, valid)
    prediction, truth = logits.float().sigmoid() * mask, target.float() * mask
    truth_sum = truth.sum(dim=(-2, -1))
    prediction_sum = prediction.sum(dim=(-2, -1))
    intersection = (prediction * truth).sum(dim=(-2, -1))
    per_class = 1 - (2 * intersection + eps) / (prediction_sum + truth_sum + eps)
    positive = truth_sum > 0
    per_image = (per_class * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return per_image.mean()


def multilabel_loss(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Masked BCE plus per-image Dice over classes with valid positive truth."""
    return masked_bce(logits, target, valid) + _positive_dice(logits, target, valid)


def _resize(logits: Tensor, size) -> Tensor:
    return logits if logits.shape[-2:] == size else F.interpolate(
        logits, size=size, mode="bilinear", align_corners=False)


def _boundary_target(target: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    truth = target.float()
    dilated = F.max_pool2d(truth, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-truth, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp_(0, 1)
    # Explicit invalid outside padding also excludes the outer image perimeter:
    # every supervised boundary pixel has an observed full3x3 neighborhood.
    invalid = F.pad((~valid.bool()).float(), (1, 1, 1, 1), value=1)
    boundary_valid = F.max_pool2d(invalid, kernel_size=3, stride=1) == 0
    return boundary, boundary_valid


def segmentation_loss(outputs: dict, target: Tensor, valid: Tensor,
                      coarse_weight: float = .2, edge_weight: float = .1) -> tuple[Tensor, dict[str, Tensor]]:
    """Main BCE+Dice, mean coarse BCE, and optional boundary BCE+Dice.

    Logits from auxiliary heads are upsampled to the target canvas; targets are
    never downsampled, which avoids erasing thin cracks from coarse supervision.
    Returned diagnostic tensors are detached so callers may retain them safely.
    """
    logits = _resize(outputs["logits"], target.shape[-2:])
    bce = masked_bce(logits, target, valid)
    dice = _positive_dice(logits, target, valid)
    main = bce + dice
    zero = logits.float().sum() * 0
    coarse = outputs.get("coarse")
    if isinstance(coarse, Tensor):
        coarse = [coarse]
    coarse_loss = torch.stack([
        masked_bce(_resize(head, target.shape[-2:]), target, valid) for head in coarse
    ]).mean() if coarse else zero
    edge_logits = outputs.get("edge")
    edge_loss = zero
    if edge_logits is not None:
        boundary, boundary_valid = _boundary_target(target, valid)
        if edge_logits.shape[1] == 1 and target.shape[1] != 1:
            boundary = boundary.amax(dim=1, keepdim=True)
            if boundary_valid.shape[1] != 1:
                boundary_valid = boundary_valid.all(dim=1, keepdim=True)
        edge_loss = multilabel_loss(_resize(edge_logits, target.shape[-2:]), boundary, boundary_valid)
    total = main + coarse_weight * coarse_loss + edge_weight * edge_loss
    parts = {"main": main, "bce": bce, "dice": dice,
             "coarse": coarse_loss, "edge": edge_loss, "total": total}
    return total, {name: value.detach() for name, value in parts.items()}
