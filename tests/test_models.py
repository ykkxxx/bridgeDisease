"""Offline model contracts: real small Transformers, no downloaded checkpoints."""
import copy
import math
import sys
from pathlib import Path

import pytest
import torch
from transformers import CLIPConfig, CLIPModel, SegformerConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bridge_seg.models import (
    BridgeTextSeg, CLIPPatchBackbone, CLIPSegBaseline, DeepLabV3Plus,
    PromptMatcher, SegFormerBaseline, load_prompts,
)


torch.set_num_threads(2)


def tiny_clip():
    config = CLIPConfig(
        text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 4,
                     "max_position_embeddings": 16, "bos_token_id": 0,
                     "eos_token_id": 2, "pad_token_id": 1},
        vision_config={"hidden_size": 32, "intermediate_size": 64,
                       "num_hidden_layers": 12, "num_attention_heads": 4,
                       "image_size": 32, "patch_size": 16},
        projection_dim=16,
    )
    return CLIPModel(config)


def cached_text():
    generator = torch.Generator().manual_seed(21)
    return torch.randn(19, 3, 16, generator=generator)


def tiny_proposed(**kwargs):
    backbone = CLIPPatchBackbone(tiny_clip(), gradient_checkpointing=True)
    return BridgeTextSeg(backbone, cached_text(), **kwargs)


def test_matching_averages_within_each_class_without_cross_class_softmax():
    # A class softmax would prevent the independent literal logits 8 and -2.
    matcher = PromptMatcher(2, adaptive=False)
    feature = torch.tensor([[[[1.0]], [[0.0]]]])
    text = torch.tensor([[[1., 0.]] * 3, [[0., 1.]] * 3])
    torch.testing.assert_close(matcher(feature, text).flatten(), torch.tensor([8., -2.]))


def test_adaptive_matching_uses_positive_temperature_and_class_local_descriptions():
    matcher = PromptMatcher(1, adaptive=True)
    feature = torch.tensor([[[[1.0]], [[0.0]]]])
    text = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]]])
    expected = 10 * ((math.exp(5) - math.exp(-5)) /
                     (math.exp(5) + 1 + math.exp(-5))) - 2
    assert matcher(feature, text).item() == pytest.approx(expected, abs=1e-5)
    with torch.no_grad():
        matcher.log_beta.fill_(100)
        matcher.log_gamma.fill_(-100)
    assert 1 <= matcher.beta.item() <= 20
    assert 1 <= matcher.gamma.item() <= 100


def test_prompt_file_validates_official_channel_order_and_repeats_class_names(tmp_path):
    import json
    prompt_path = Path(__file__).resolve().parents[1] / "docs/thesis/class_prompts_draft.json"
    description, names = load_prompts(prompt_path)
    assert len(description) == len(names) == 19
    assert description[0][1] == "a thin irregular fracture line in concrete"
    assert names[0] == ["a photo of a crack on a concrete surface"] * 3
    invalid = json.loads(prompt_path.read_text(encoding="utf-8"))
    invalid["classes"][0]["label"] = "Spalling"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="channel|order|label"):
        load_prompts(path)


@pytest.mark.parametrize("local,edge,adaptive", [(False, False, False),
    (False, False, True), (True, False, True), (True, True, True)])
def test_proposed_preserves_odd_geometry_and_propagates_decoder_gradients(local, edge, adaptive):
    model = tiny_proposed(local_branch=local, boundary_head=edge, adaptive=adaptive)
    outputs = model(torch.rand(1, 3, 35, 41))
    assert outputs["logits"].shape == (1, 19, 35, 41)
    assert len(outputs["coarse"]) == 4
    assert all(x.shape == (1, 19, 35, 41) for x in outputs["coarse"])
    assert ("edge" in outputs) == edge
    loss = outputs["logits"].square().mean()
    loss = loss + sum(x.square().mean() for x in outputs["coarse"])
    if edge:
        assert outputs["edge"].shape == (1, 19, 35, 41)
        loss = loss + outputs["edge"].square().mean()
    loss.backward()
    assert model.pixel_projection.weight.grad is not None
    assert torch.isfinite(model.pixel_projection.weight.grad).all()
    assert not model.text_embeddings.requires_grad
    assert all(p.grad is None for p in model.backbone.parameters())
    model.eval()
    with torch.no_grad():
        assert set(model(torch.rand(1, 3, 32, 32))) == {"logits"}


def test_final_two_clip_blocks_unfreeze_at_epoch_six_and_reach_optimizer():
    model = tiny_proposed()
    candidates = list(model.backbone_parameters())
    assert candidates and not any(p.requires_grad for p in candidates)
    optimizer = torch.optim.AdamW([{"params": candidates},
        {"params": [p for p in model.parameters() if p.requires_grad]}])
    model.set_train_epoch(5)
    assert not any(p.requires_grad for p in candidates)
    model.set_train_epoch(6)
    assert all(p.requires_grad for p in candidates)
    output = model(torch.rand(1, 3, 32, 32))["logits"]
    output.square().mean().backward()
    last_layers = model.backbone.vision_model.encoder.layers
    assert last_layers[-1].self_attn.q_proj.weight.grad is not None
    assert last_layers[-3].self_attn.q_proj.weight.grad is None
    assert model.backbone.visual_projection.weight.grad is None
    before = candidates[0].detach().clone()
    optimizer.step()
    assert not torch.equal(before, candidates[0])


def test_patch_backbone_uses_layers_three_six_nine_twelve_and_rectangular_positions():
    clip = tiny_clip().eval()
    model = CLIPPatchBackbone(clip).eval()
    image = torch.rand(1, 3, 32, 48)
    mean = torch.tensor([.48145466, .4578275, .40821073]).view(1, 3, 1, 1)
    std = torch.tensor([.26862954, .26130258, .27577711]).view(1, 3, 1, 1)
    with torch.no_grad():
        reference = clip.vision_model((image - mean) / std,
            output_hidden_states=True, interpolate_pos_encoding=True)
        actual = model(image)
        for feature, layer in zip(actual, (3, 6, 9, 12)):
            tokens = clip.visual_projection(clip.vision_model.post_layernorm(
                reference.hidden_states[layer][:, 1:]))
            expected = tokens.transpose(1, 2).reshape(1, 16, 2, 3)
            torch.testing.assert_close(feature, expected)


def test_learned_prototypes_receive_gradients_and_match_decoder_architecture():
    model = tiny_proposed(learned_prototypes=True)
    assert model.prototypes.shape == (19, 3, 16)
    model(torch.rand(1, 3, 32, 32))["logits"].square().mean().backward()
    assert model.prototypes.grad is not None
    assert torch.isfinite(model.prototypes.grad).all()


def test_clipseg_shared_vision_and_checkpointed_class_gradients_match_direct():
    model = CLIPSegBaseline(tiny_clip(), cached_text()[:, 0],
                            gradient_checkpointing=True, reduce_dim=8)
    direct = copy.deepcopy(model)
    direct.gradient_checkpointing = False
    counts = []
    hook = model.vision_model.register_forward_hook(lambda *_: counts.append(1))
    image = torch.rand(1, 3, 35, 41)
    checkpointed_logits = model(image)["logits"]
    direct_logits = direct(image)["logits"]
    assert counts == [1]
    hook.remove()
    assert checkpointed_logits.shape == (1, 19, 35, 41)
    torch.testing.assert_close(checkpointed_logits, direct_logits)
    checkpointed_logits.square().mean().backward()
    direct_logits.square().mean().backward()
    for parameter, reference in zip(model.decoder.parameters(), direct.decoder.parameters()):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference.grad)
    assert all(p.grad is None for p in model.vision_model.parameters())
    assert list(model.backbone_parameters()) == []


def test_segformer_random_head_runs_at_input_geometry():
    config = SegformerConfig(hidden_sizes=[8, 16, 24, 32], depths=[1, 1, 1, 1],
        num_attention_heads=[1, 2, 3, 4], sr_ratios=[8, 4, 2, 1],
        decoder_hidden_size=16, num_labels=19)
    model = SegFormerBaseline(config=config, pretrained=False)
    output = model(torch.rand(1, 3, 65, 73))["logits"]
    assert output.shape == (1, 19, 65, 73)
    output.mean().backward()
    assert model.model.decode_head.classifier.weight.grad is not None


def test_deeplab_batchnorm_statistics_remain_frozen_for_microbatch_one():
    model = DeepLabV3Plus(pretrained=False)
    model.train()
    batchnorm = next(m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = batchnorm.running_mean.clone()
    output = model(torch.rand(1, 3, 33, 41))["logits"]
    assert output.shape == (1, 19, 33, 41)
    output.mean().backward()
    torch.testing.assert_close(batchnorm.running_mean, before)
    assert batchnorm.weight.grad is not None
