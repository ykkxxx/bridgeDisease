"""Supervised, independent-channel segmentation models for the locked experiment IDs.

Inputs are raw RGB floats in [0, 1]. Only backbone weights are pretrained;
segmentation heads are always initialized here. B5/CAT-Seg is deliberately absent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .constants import CLASS_NAMES, CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD


def load_prompts(path: str | Path) -> tuple[list[list[str]], list[list[str]]]:
    """Validate and preserve the dataset's official channel order."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    classes = payload.get("classes", [])
    if len(classes) != len(CLASS_NAMES):
        raise ValueError("Prompt file must contain all 19 classes in channel order")
    descriptions, names = [], []
    for channel, (entry, label) in enumerate(zip(classes, CLASS_NAMES)):
        if entry.get("channel") != channel or entry.get("label") != label:
            raise ValueError(f"Prompt channel {channel} must have label {label}")
        prompts = entry.get("descriptions", [])
        name = entry.get("class_name_prompt")
        if len(prompts) != 3 or not all(isinstance(s, str) and s.strip() for s in prompts):
            raise ValueError(f"{label} needs three nonempty descriptions")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} needs a nonempty class_name_prompt")
        descriptions.append(prompts)
        names.append([name] * 3)
    return descriptions, names


class InputNormalization(nn.Module):
    def __init__(self, mean: tuple, std: tuple):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: Tensor) -> Tensor:
        return (image - self.mean) / self.std


def _resize(value: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


def _pad_to_patch(image: Tensor, patch_size: int, square: bool = False) -> Tensor:
    height, width = image.shape[-2:]
    padded_h = math.ceil(height / patch_size) * patch_size
    padded_w = math.ceil(width / patch_size) * patch_size
    if square:
        padded_h = padded_w = max(padded_h, padded_w)
    return F.pad(image, (0, padded_w - width, 0, padded_h - height), mode="replicate")


class ConvNormAct(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int, kernel: int = 3,
                 stride: int = 1, dilation: int = 1):
        super().__init__(
            nn.Conv2d(incoming, outgoing, kernel, stride=stride,
                      padding=dilation * (kernel // 2), dilation=dilation, bias=False),
            nn.GroupNorm(math.gcd(8, outgoing), outgoing), nn.GELU(),
        )


class PromptMatcher(nn.Module):
    """Cosines and a softmax over each class's own descriptions, never classes."""
    def __init__(self, num_classes: int = 19, adaptive: bool = True):
        super().__init__()
        self.adaptive = adaptive
        self.log_beta = nn.Parameter(torch.tensor(math.log(5.0)), requires_grad=adaptive)
        self.log_gamma = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.full((num_classes,), -2.0))

    @property
    def beta(self) -> Tensor:
        return self.log_beta.clamp(0, math.log(20)).exp()

    @property
    def gamma(self) -> Tensor:
        return self.log_gamma.clamp(0, math.log(100)).exp()

    def forward(self, features: Tensor, text: Tensor) -> Tensor:
        # Keep normalization and matching statistics in FP32 under mixed precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = F.normalize(features.float(), dim=1, eps=1e-6)
            text = F.normalize(text.float(), dim=-1, eps=1e-6)
            cosine = torch.einsum("bdhw,kmd->bkmhw", features, text)
            if self.adaptive:
                weights = (self.beta * cosine).softmax(dim=2)
                matched = (weights * cosine).sum(dim=2)
            else:
                matched = cosine.mean(dim=2)
            return self.gamma * matched + self.bias[None, :, None, None]


class CLIPPatchBackbone(nn.Module):
    """CLIP spatial tokens with exact block indices and staged vision gradients.

    Construct with a real HF CLIPModel, including a tiny random config for offline
    tests. The text encoder is not retained after its fixed embeddings are cached.
    """
    def __init__(self, clip: nn.Module, extract_layers: tuple[int, ...] = (3, 6, 9, 12),
                 gradient_checkpointing: bool = True):
        super().__init__()
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection
        self.extract_layers = tuple(extract_layers)
        self.embed_dim = clip.config.projection_dim
        self.patch_size = clip.config.vision_config.patch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.normalization = InputNormalization(CLIP_MEAN, CLIP_STD)
        if max(self.extract_layers) > len(self.vision_model.encoder.layers):
            raise ValueError("Requested CLIP feature layer exceeds the encoder depth")
        self.requires_grad_(False)
        self.set_train_epoch(1)

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for layer in self.vision_model.encoder.layers[-2:]:
            yield from layer.parameters()

    def set_train_epoch(self, epoch: int):
        self.unfrozen = epoch >= 6
        for parameter in self.backbone_parameters():
            parameter.requires_grad_(self.unfrozen)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        if getattr(self, "unfrozen", False):
            for layer in self.vision_model.encoder.layers[-2:]:
                layer.train(mode)
        return self

    def forward(self, image: Tensor) -> list[Tensor]:
        image = self.normalization(image)
        height, width = image.shape[-2:]
        with torch.no_grad():
            hidden = self.vision_model.embeddings(image, interpolate_pos_encoding=True)
            hidden = self.vision_model.pre_layrnorm(hidden)
        features = []
        for index, layer in enumerate(self.vision_model.encoder.layers, start=1):
            trainable = any(p.requires_grad for p in layer.parameters())
            if not trainable:
                with torch.no_grad():
                    hidden = layer(hidden, attention_mask=None, causal_attention_mask=None)[0]
            elif self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                def run(value, block=layer):
                    return block(value, attention_mask=None, causal_attention_mask=None)[0]
                hidden = checkpoint(run, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden, attention_mask=None, causal_attention_mask=None)[0]
            if index in self.extract_layers:
                # Frozen projection/LN still transmit gradients from the final blocks.
                patches = self.visual_projection(self.vision_model.post_layernorm(hidden[:, 1:]))
                features.append(patches.transpose(1, 2).reshape(
                    image.shape[0], self.embed_dim, height // self.patch_size, width // self.patch_size))
        return features


class ResidualAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 64):
        super().__init__()
        # 1x1 convolutions implement the same per-token Linear layers.
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: Tensor) -> Tensor:
        return F.normalize(feature + self.up(F.gelu(self.down(feature))), dim=1, eps=1e-6)


class LocalDetails(nn.Module):
    def __init__(self):
        super().__init__()
        self.l4 = nn.Sequential(ConvNormAct(3, 32, stride=2), ConvNormAct(32, 64, stride=2),
                                ConvNormAct(64, 64))
        self.l8 = nn.Sequential(ConvNormAct(64, 128, stride=2), ConvNormAct(128, 128))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        l4 = self.l4(image)
        return l4, self.l8(l4)


class BridgeTextSeg(nn.Module):
    """Proposed model and strictly matched learned-prototype/prompt ablations."""
    def __init__(self, backbone: CLIPPatchBackbone, text_embeddings: Tensor | None = None,
                 *, local_branch: bool = True, boundary_head: bool = True,
                 adaptive: bool = True, learned_prototypes: bool = False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = len(CLASS_NAMES)
        if len(backbone.extract_layers) != 4:
            raise ValueError("BridgeTextSeg requires four CLIP feature layers")
        embedding_dim = backbone.embed_dim
        shape = (self.num_classes, 3, embedding_dim)
        if learned_prototypes:
            self.prototypes = nn.Parameter(torch.randn(shape) / math.sqrt(embedding_dim))
            self.register_buffer("text_embeddings", torch.empty(0))
        else:
            if text_embeddings is None or tuple(text_embeddings.shape) != shape:
                raise ValueError(f"Expected cached text embeddings with shape {shape}")
            self.register_buffer("text_embeddings", F.normalize(text_embeddings.detach().float(), dim=-1))
            self.register_parameter("prototypes", None)
        self.adapters = nn.ModuleList([ResidualAdapter(embedding_dim) for _ in range(4)])
        self.reductions = nn.ModuleList([nn.Conv2d(embedding_dim, 128, 1) for _ in range(4)])
        self.coarse_matchers = nn.ModuleList([PromptMatcher(self.num_classes, adaptive) for _ in range(4)])
        self.semantic_fusion = ConvNormAct(512 + 4 * self.num_classes, 256, kernel=1)
        self.local = LocalDetails() if local_branch else None
        self.decode8 = nn.Sequential(ConvNormAct(384 if local_branch else 256, 256), ConvNormAct(256, 256))
        self.decode4 = nn.Sequential(ConvNormAct(320 if local_branch else 256, 128), ConvNormAct(128, 128))
        self.pixel_projection = nn.Conv2d(128, embedding_dim, 1)
        self.final_matcher = PromptMatcher(self.num_classes, adaptive)
        self.edge_head = nn.Conv2d(128, self.num_classes, 1) if boundary_head else None

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.backbone_parameters()

    def set_train_epoch(self, epoch: int):
        self.backbone.set_train_epoch(epoch)

    def forward(self, image: Tensor) -> dict[str, Tensor | list[Tensor]]:
        original_size = image.shape[-2:]
        image = _pad_to_patch(image, self.backbone.patch_size)
        padded_size = image.shape[-2:]
        text = self.prototypes if self.prototypes is not None else self.text_embeddings
        adapted = [adapter(feature) for adapter, feature in zip(self.adapters, self.backbone(image))]
        coarse = [matcher(feature, text) for matcher, feature in zip(self.coarse_matchers, adapted)]
        reduced = [reduction(feature) for reduction, feature in zip(self.reductions, adapted)]
        semantic = self.semantic_fusion(torch.cat(reduced + coarse, dim=1))
        if self.local is not None:
            l4, l8 = self.local(image)
            f8 = self.decode8(torch.cat((_resize(semantic, l8.shape[-2:]), l8), dim=1))
            f4 = self.decode4(torch.cat((_resize(f8, l4.shape[-2:]), l4), dim=1))
        else:
            h, w = padded_size
            f8 = self.decode8(_resize(semantic, (h // 8, w // 8)))
            f4 = self.decode4(_resize(f8, (h // 4, w // 4)))
        logits = self.final_matcher(self.pixel_projection(f4), text)
        height, width = original_size
        def restore(value):
            return _resize(value, padded_size)[..., :height, :width]
        outputs = {"logits": restore(logits)}
        if self.training:
            outputs["coarse"] = [restore(value) for value in coarse]
            if self.edge_head is not None:
                outputs["edge"] = restore(self.edge_head(f4))
        return outputs


class CLIPSegBaseline(nn.Module):
    """HF 4.46.3 CLIPSeg decoder, random initialization, shared frozen CLIP.

    The official decoder requires a square token grid, so non-square images are
    padded rather than stretched. Each class is checkpointed independently;
    summed multi-label loss retains exact gradients while recomputing activations.
    Source: transformers/v4.46.3/.../clipseg/modeling_clipseg.py.
    No CIDAS/PhraseCut segmentation checkpoint is loaded.
    """
    def __init__(self, clip: nn.Module, class_embeddings: Tensor,
                 gradient_checkpointing: bool = True, reduce_dim: int = 64):
        super().__init__()
        from transformers import CLIPSegConfig
        from transformers.models.clipseg.modeling_clipseg import CLIPSegDecoder
        self.vision_model = clip.vision_model
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()
        self.patch_size = clip.config.vision_config.patch_size
        self.extract_layers = (3, 6, 9)  # original CLIPSeg zero-based blocks 4/7/10
        self.gradient_checkpointing = gradient_checkpointing
        config = CLIPSegConfig(
            text_config=clip.config.text_config.to_dict(),
            vision_config=clip.config.vision_config.to_dict(),
            projection_dim=clip.config.projection_dim, extract_layers=list(self.extract_layers),
            reduce_dim=reduce_dim, decoder_num_attention_heads=4,
            conditional_layer=0, use_complex_transposed_convolution=True,
        )
        self.decoder = CLIPSegDecoder(config)
        if tuple(class_embeddings.shape) != (len(CLASS_NAMES), clip.config.projection_dim):
            raise ValueError("CLIPSeg requires one cached class-name embedding per output class")
        # Original CLIPSeg FiLM consumes unnormalized projected text features.
        self.register_buffer("class_embeddings", class_embeddings.detach().float().clone())
        self.normalization = InputNormalization(CLIP_MEAN, CLIP_STD)

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return iter(())

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        return self

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        height, width = image.shape[-2:]
        image = _pad_to_patch(image, self.patch_size, square=True)
        with torch.no_grad():
            encoded = self.vision_model(self.normalization(image), output_hidden_states=True,
                                       interpolate_pos_encoding=True, return_dict=True)
            features = tuple(encoded.hidden_states[layer + 1] for layer in self.extract_layers)
        predictions = []
        for class_embedding in self.class_embeddings:
            condition = class_embedding.unsqueeze(0).expand(image.shape[0], -1)
            def decode(*values):
                return self.decoder(values[:-1], values[-1], return_dict=True).logits
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                prediction = checkpoint(decode, *features, condition, use_reentrant=False)
            else:
                prediction = decode(*features, condition)
            predictions.append(prediction.unsqueeze(1))
        logits = _resize(torch.cat(predictions, dim=1), image.shape[-2:])
        return {"logits": logits[..., :height, :width]}


class ASPP(nn.Module):
    def __init__(self, incoming: int = 2048, outgoing: int = 256):
        super().__init__()
        self.branches = nn.ModuleList([ConvNormAct(incoming, outgoing, kernel=1)] +
            [ConvNormAct(incoming, outgoing, dilation=rate) for rate in (6, 12, 18)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvNormAct(incoming, outgoing, kernel=1))
        self.fuse = ConvNormAct(5 * outgoing, outgoing, kernel=1)

    def forward(self, value: Tensor) -> Tensor:
        branches = [branch(value) for branch in self.branches]
        branches.append(_resize(self.pool(value), value.shape[-2:]))
        return self.fuse(torch.cat(branches, dim=1))


class DeepLabV3Plus(nn.Module):
    """ImageNet ResNet50 + ASPP + low-level V3+ fusion; no segmentation weights."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
                                 replace_stride_with_dilation=[False, False, True])
        self.backbone.fc = nn.Identity()
        self.aspp = ASPP()
        self.low_projection = ConvNormAct(256, 48, kernel=1)
        self.decoder = nn.Sequential(ConvNormAct(304, 256), ConvNormAct(256, 256),
                                     nn.Conv2d(256, len(CLASS_NAMES), 1))
        self.normalization = InputNormalization(IMAGENET_MEAN, IMAGENET_STD)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # affine parameters remain trainable
        return self

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        original_size = image.shape[-2:]
        value = self.normalization(image)
        b = self.backbone
        value = b.maxpool(b.relu(b.bn1(b.conv1(value))))
        low = b.layer1(value)
        high = b.layer4(b.layer3(b.layer2(low)))
        fused = torch.cat((_resize(self.aspp(high), low.shape[-2:]), self.low_projection(low)), dim=1)
        return {"logits": _resize(self.decoder(fused), original_size)}


class SegFormerBaseline(nn.Module):
    """ImageNet MiT-B2 encoder and a new 19-channel SegFormer decode head."""
    def __init__(self, model_name: str = "nvidia/mit-b2", pretrained: bool = True, config=None):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel
        if config is None:
            if pretrained:
                config = SegformerConfig.from_pretrained(model_name)
            else:
                config = SegformerConfig(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
                    num_attention_heads=[1, 2, 5, 8], decoder_hidden_size=256)
        config.num_labels = len(CLASS_NAMES)
        config.id2label = dict(enumerate(CLASS_NAMES))
        config.label2id = {label: index for index, label in enumerate(CLASS_NAMES)}
        self.model = SegformerForSemanticSegmentation(config)
        if pretrained:
            # Loading SegformerModel never loads a task-specific segmentation head.
            self.model.segformer = SegformerModel.from_pretrained(model_name, config=config)
        self.normalization = InputNormalization(IMAGENET_MEAN, IMAGENET_STD)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        logits = self.model(pixel_values=self.normalization(image), return_dict=True).logits
        return {"logits": _resize(logits, image.shape[-2:])}


def _cache_text(clip: nn.Module, prompt_groups: list[list[str]], model_name: str,
                local_files_only: bool, normalize: bool = True) -> Tensor:
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    prompts = [text for group in prompt_groups for text in group]
    tokens = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    if tokens["input_ids"].shape[1] > clip.config.text_config.max_position_embeddings:
        raise ValueError("A prompt exceeds CLIP's maximum token length; shorten it without truncation")
    clip.eval()
    clip.requires_grad_(False)
    with torch.no_grad():
        embeddings = clip.get_text_features(**tokens).float()
    if normalize:
        embeddings = F.normalize(embeddings, dim=-1, eps=1e-6)
    return embeddings.reshape(len(prompt_groups), len(prompt_groups[0]), -1)


def build_model(model_id: str, prompt_file: str | Path,
                clip_model_name: str = "openai/clip-vit-base-patch16",
                segformer_model_name: str = "nvidia/mit-b2", pretrained: bool = True,
                gradient_checkpointing: bool = True) -> nn.Module:
    """Create an experiment model; ``pretrained=False`` never downloads assets.

    Text variants still require a cached/local *real CLIP tokenizer* in that mode.
    For fully offline small tests use BridgeTextSeg with injected cached embeddings
    and a CLIPPatchBackbone built from a tiny CLIPConfig.
    """
    model_id = model_id.upper()
    if model_id == "A5":
        model_id = "P"
    if model_id == "B1":
        return DeepLabV3Plus(pretrained=pretrained)
    if model_id == "B2":
        return SegFormerBaseline(segformer_model_name, pretrained=pretrained)
    variants = {"P": (True, True, True), "B3": (True, True, True),
                "A0": (False, False, False), "A1": (False, False, False),
                "A2": (False, False, True), "A3": (True, False, False),
                "A4": (True, False, True), "A1L": (True, False, False)}
    if model_id not in variants and model_id != "B4":
        raise ValueError(f"Unknown model {model_id}; supported: B1-B4, P, A0-A5, A1L. CAT-Seg/B5 is not implemented.")
    from transformers import CLIPConfig, CLIPModel
    if pretrained:
        clip = CLIPModel.from_pretrained(clip_model_name)
    else:
        local_config = Path(clip_model_name) / "config.json"
        config = CLIPConfig.from_pretrained(clip_model_name, local_files_only=True) if local_config.exists() else CLIPConfig(
            vision_config={"patch_size": 16})
        clip = CLIPModel(config)
    descriptions, names = load_prompts(prompt_file)
    if model_id == "B4":
        cached = _cache_text(clip, [[x[0]] for x in names], clip_model_name,
                             local_files_only=not pretrained, normalize=False)
        return CLIPSegBaseline(clip, cached[:, 0], gradient_checkpointing=gradient_checkpointing)
    local, boundary, adaptive = variants[model_id]
    text = None
    if model_id != "B3":
        prompts = names if model_id in {"A0", "A3"} else descriptions
        text = _cache_text(clip, prompts, clip_model_name, local_files_only=not pretrained)
    backbone = CLIPPatchBackbone(clip, gradient_checkpointing=gradient_checkpointing)
    return BridgeTextSeg(backbone, text, local_branch=local, boundary_head=boundary,
                         adaptive=adaptive, learned_prototypes=model_id == "B3")
