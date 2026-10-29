"""Prepared dacl10k images and independent polygon masks.

Coordinates use the decoded image canvas (without EXIF auto-rotation). Each
filled ring is intersected with the unit canvas before rasterization. Pillow
fills the resulting polygon at continuous coordinates ``(x * W, y * H)``;
its inclusive polygon boundary rule is shared by every class. Multiple rings
and instances are unioned within a class; different classes retain overlap.
"""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from .constants import CLASS_NAMES


def _relative_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Image filepath must be a nonempty relative path")
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(value).drive or ".." in relative.parts:
        raise ValueError(f"Unsafe image filepath: {value!r}")
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image filepath escapes data_root: {value!r}")
    return path


def _clip_ring(ring: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman intersection with [0,1] x [0,1].

    Intersection vertices are computed on the original edges. Independently
    clamping outside vertices would change the shape visible inside the image.
    """
    polygon = []
    for point in ring:
        if len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not math.isfinite(v) for v in point):
            raise ValueError("Polygon coordinates must be finite numeric (x, y) pairs")
        polygon.append((float(point[0]), float(point[1])))
    if len(polygon) < 3:
        raise ValueError("A polygon ring needs at least three vertices")
    for axis, boundary, keep_greater in ((0, 0., True), (0, 1., False),
                                         (1, 0., True), (1, 1., False)):
        if not polygon:
            break
        clipped = []
        previous = polygon[-1]
        previous_inside = previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        for current in polygon:
            inside = current[axis] >= boundary if keep_greater else current[axis] <= boundary
            if inside != previous_inside:
                fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
                intersection = [previous[0] + fraction * (current[0] - previous[0]),
                                previous[1] + fraction * (current[1] - previous[1])]
                intersection[axis] = boundary
                clipped.append(tuple(intersection))
            if inside:
                clipped.append(current)
            previous, previous_inside = current, inside
        polygon = clipped
    return polygon


def _rasterize(regions: Sequence[dict], width: int, height: int) -> torch.Tensor:
    # Allocate uint8 while drawing; convert to floating point only after crop.
    masks = np.zeros((len(CLASS_NAMES), height, width), dtype=np.uint8)
    class_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    canvases: dict[int, Image.Image] = {}
    for region in regions:
        label = region.get("label")
        if label not in class_index:
            raise ValueError(f"Unknown polygon label: {label!r}")
        if region.get("closed") is not True or region.get("filled") is not True:
            raise ValueError("Expected a closed filled polygon")
        rings = region.get("points")
        if not isinstance(rings, list) or not rings:
            raise ValueError("Polygon points must contain at least one ring")
        index = class_index[label]
        if index not in canvases:
            canvases[index] = Image.new("L", (width, height), 0)
        canvas = canvases[index]
        draw = ImageDraw.Draw(canvas)
        for ring in rings:
            polygon = _clip_ring(ring)
            if len(set(polygon)) < 3:
                continue
            if max(p[0] for p in polygon) <= min(p[0] for p in polygon) or \
                    max(p[1] for p in polygon) <= min(p[1] for p in polygon):
                continue
            draw.polygon([(x * width, y * height) for x, y in polygon], fill=1)
    for index, canvas in canvases.items():
        masks[index] = np.asarray(canvas, dtype=np.uint8)
    return torch.from_numpy(masks)


def _randint(maximum: int) -> int:
    return int(torch.randint(maximum + 1, ()).item()) if maximum > 0 else 0


class BridgeDataset(Dataset):
    """Load only ``samples_clean.json`` and a prepared split JSONL manifest.

    Training resizes the longest side to ``max_long_side * U(scale_range)``,
    then takes a random square crop with zero padding and shared horizontal /
    vertical flips. Randomness comes from torch, so DataLoader worker seeding
    and the runner seed govern these transforms.

    Evaluation with ``image_size=None`` preserves original image geometry for
    full-image sliding evaluation. A finite size gives an aspect-preserving
    resize and centered square padding, useful for compact deterministic tests.
    Unlabeled manifests are rejected: testdev images must use inference I/O.
    """

    def __init__(self, data_root, prepared_root, split="train", image_size=512,
                 training=False, max_long_side=1536, scale_range=(.75, 1.25)):
        self.data_root = Path(data_root).resolve()
        self.prepared_root = Path(prepared_root).resolve()
        self.image_size = image_size
        self.training = bool(training)
        self.max_long_side = max_long_side
        if image_size is not None and (isinstance(image_size, bool) or
                                       not isinstance(image_size, int) or image_size <= 0):
            raise ValueError("image_size must be a positive integer or None")
        if self.training and image_size is None:
            raise ValueError("Training requires a finite crop image_size")
        if isinstance(max_long_side, bool) or not isinstance(max_long_side, int) or max_long_side <= 0:
            raise ValueError("max_long_side must be a positive integer")
        if len(scale_range) != 2 or not all(math.isfinite(x) and x > 0 for x in scale_range) \
                or scale_range[0] > scale_range[1]:
            raise ValueError("scale_range must be two ordered positive finite numbers")
        self.scale_range = tuple(float(x) for x in scale_range)
        if not isinstance(split, str) or not split or Path(split).name != split or "\\" in split or "/" in split:
            raise ValueError("split must be a manifest basename")
        manifest = self.prepared_root / "manifests" / f"{split}.jsonl"
        with manifest.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        with (self.prepared_root / "samples_clean.json").open(encoding="utf-8") as stream:
            export = json.load(stream)
        samples = export.get("samples") if isinstance(export, dict) else None
        if not isinstance(samples, list):
            raise ValueError("samples_clean.json must contain a samples array")
        self.records = []
        for row in rows:
            if not isinstance(row, dict) or row.get("annotation_available") is not True:
                raise ValueError("Training/evaluation requires labeled records; use inference for testdev")
            if row.get("annotation_file", "samples_clean.json") != "samples_clean.json":
                raise ValueError("Manifest must reference prepared samples_clean.json")
            index = row.get("annotation_index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(samples):
                raise ValueError("Manifest annotation_index is outside samples_clean.json")
            sample = samples[index]
            if not isinstance(sample, dict) or sample.get("filepath") != row.get("filepath"):
                raise ValueError("Manifest filepath does not match its prepared annotation index")
            annotation = sample.get("semseg_ground_truth_filled")
            if not isinstance(annotation, dict) or not isinstance(annotation.get("polylines"), list):
                raise ValueError("Labeled record is missing its polygon annotation")
            path = _relative_path(self.data_root, row.get("filepath"))
            self.records.append((row, path, annotation["polylines"]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row, path, regions = self.records[index]
        with Image.open(path) as source:
            # The polygon canvas is the stored raster. EXIF transpose here would
            # rotate just the RGB image and silently misalign the annotations.
            image = source.convert("RGB")
        width, height = image.size
        if self.image_size is not None:
            longest = float(self.image_size)
            if self.training:
                low, high = self.scale_range
                random_scale = low + (high - low) * float(torch.rand(()).item())
                longest = self.max_long_side * random_scale
            scale = longest / max(width, height)
            width, height = max(1, round(width * scale)), max(1, round(height * scale))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1)
        # Rasterize at resized geometry to avoid a full original-size 19-channel
        # float mask in each worker. Normalized polygon geometry is unchanged.
        target = _rasterize(regions, width, height)
        valid = torch.ones((1, height, width), dtype=torch.bool)
        if self.image_size is not None:
            size = self.image_size
            canvas_h, canvas_w = max(size, height), max(size, width)
            gap_h, gap_w = canvas_h - height, canvas_w - width
            top = _randint(gap_h) if self.training else gap_h // 2
            left = _randint(gap_w) if self.training else gap_w // 2
            if gap_h or gap_w:
                image_pad = torch.zeros((3, canvas_h, canvas_w), dtype=image_tensor.dtype)
                target_pad = torch.zeros((len(CLASS_NAMES), canvas_h, canvas_w), dtype=target.dtype)
                valid_pad = torch.zeros((1, canvas_h, canvas_w), dtype=torch.bool)
                image_pad[:, top:top + height, left:left + width] = image_tensor
                target_pad[:, top:top + height, left:left + width] = target
                valid_pad[:, top:top + height, left:left + width] = valid
                image_tensor, target, valid = image_pad, target_pad, valid_pad
            y = _randint(canvas_h - size) if self.training else (canvas_h - size) // 2
            x = _randint(canvas_w - size) if self.training else (canvas_w - size) // 2
            image_tensor = image_tensor[:, y:y + size, x:x + size]
            target = target[:, y:y + size, x:x + size]
            valid = valid[:, y:y + size, x:x + size]
            if self.training:
                for dimension in (1, 2):
                    if torch.rand(()).item() < .5:
                        image_tensor = image_tensor.flip(dimension)
                        target = target.flip(dimension)
                        valid = valid.flip(dimension)
        return {"image": image_tensor.to(dtype=torch.float32).div_(255),
                "target": target.to(dtype=torch.float32), "valid": valid,
                "image_id": str(row.get("image_id", row["filepath"])),
                "filepath": row["filepath"]}
