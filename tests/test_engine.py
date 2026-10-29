import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridge_seg.engine import sliding_window_predict, train_one_epoch
from bridge_seg.losses import multilabel_loss
from bridge_seg.runner import load_config, stage_splits


class PixelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 19, 1)

    def forward(self, image):
        return {"logits": self.conv(image)}


class EngineTests(unittest.TestCase):
    def test_final_stage_does_not_have_a_validation_or_test_loader(self):
        self.assertEqual(stage_splits("dev"), ("train", "val"))
        self.assertEqual(stage_splits("final"), ("final_train", None))

    def test_config_inheritance_defaults_to_40_epochs(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/experiments/P.json")
        self.assertEqual(cfg["epochs"], 40)
        self.assertEqual(cfg["model"], "P")
        self.assertEqual(cfg["image_size"], 512)

    def test_sliding_predictions_preserve_original_geometry(self):
        torch.manual_seed(7)
        model = PixelModel().eval()
        image = torch.rand(3, 22, 31)
        expected = model(image.unsqueeze(0))["logits"].detach().cpu()
        actual = sliding_window_predict(model, image, torch.device("cpu"), image_size=16,
                                        stride=11, max_long_side=100, amp=False)
        self.assertEqual(tuple(actual.shape), (1, 19, 22, 31))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_partial_accumulation_group_has_correct_scale(self):
        torch.manual_seed(11)
        model, reference = PixelModel(), PixelModel()
        reference.load_state_dict(model.state_dict())
        samples = [{"image": torch.rand(3, 8, 8), "target": torch.randint(0, 2, (19, 8, 8)).float(),
                    "valid": torch.ones(1, 8, 8, dtype=torch.bool)} for _ in range(3)]
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        result = train_one_epoch(model, DataLoader(samples, batch_size=1), optimizer,
                                 torch.device("cpu"), accumulation_steps=2, amp=False, grad_clip=0)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
        for group in (samples[:2], samples[2:]):
            expected_optimizer.zero_grad()
            images = torch.stack([sample["image"] for sample in group])
            target = torch.stack([sample["target"] for sample in group])
            valid = torch.stack([sample["valid"] for sample in group])
            multilabel_loss(reference(images)["logits"], target, valid).backward()
            expected_optimizer.step()
        self.assertEqual(result["optimizer_steps"], 2)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
