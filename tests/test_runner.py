import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_seg.runner import train_main, _checkpoint_state, _load_weights, _frozen_fingerprint


class TinySegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 19, 1)

    def forward(self, image):
        return {"logits": self.head(image)}


class RunnerSmokeTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "data" / ".test-tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw, self.prepared = self.root / "raw", self.root / "prepared"
        self.raw.mkdir()
        (self.prepared / "manifests").mkdir(parents=True)
        samples, rows = [], []
        for index in range(4):
            filename = f"{index}.png"
            Image.new("RGB", (35, 41), (70 + index * 10, 100, 140)).save(self.raw / filename)
            samples.append({"_id": {"$oid": str(index)}, "filepath": filename,
                            "semseg_ground_truth_filled": {"polylines": [{"label": "Crack",
                            "closed": True, "filled": True, "points": [[[.1, .1], [.8, .1], [.5, .8]]]}]}})
            rows.append({"image_id": str(index), "filepath": filename, "annotation_index": index,
                         "annotation_available": True, "width": 35, "height": 41})
        (self.prepared / "samples_clean.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")
        for split, indices in (("train", [0, 1]), ("val", [2]), ("test", [3]), ("final_train", [0, 1, 2])):
            (self.prepared / f"manifests/{split}.jsonl").write_text(
                "".join(json.dumps({**rows[index], "split": split}) + "\n" for index in indices), encoding="utf-8")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"_base": str(ROOT / "configs/train_base.json"), "model": "P",
            "image_size": 32, "epochs": 1, "num_workers": 0, "amp": False,
            "max_long_side": 48, "data_root": str(self.raw), "prepared_root": str(self.prepared)}), encoding="utf-8")

    def run_training(self, stage, run_dir, *args):
        with patch("bridge_seg.runner._model", side_effect=lambda cfg: TinySegmentation()):
            return train_main(["--config", str(self.config), "--device", "cpu", "--stage", stage,
                               "--run-dir", str(run_dir), *args])

    def test_development_saves_resumable_last_and_selected_best(self):
        run_dir = self.root / "dev"
        self.assertEqual(self.run_training("dev", run_dir), 0)
        self.assertTrue((run_dir / "last.pt").exists())
        self.assertTrue((run_dir / "best.pt").exists())
        self.assertFalse((run_dir / "final.pt").exists())
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertIn("optimizer", checkpoint)
        self.assertIn("rng", checkpoint)
        record = json.loads((run_dir / "history.jsonl").read_text(encoding="utf-8"))
        self.assertIn("val", record)

    def test_final_training_never_reads_val_or_test(self):
        (self.prepared / "manifests/val.jsonl").unlink()
        (self.prepared / "manifests/test.jsonl").unlink()
        run_dir = self.root / "final"
        self.assertEqual(self.run_training("final", run_dir), 0)
        self.assertTrue((run_dir / "final.pt").exists())
        self.assertFalse((run_dir / "best.pt").exists())
        record = json.loads((run_dir / "history.jsonl").read_text(encoding="utf-8"))
        self.assertNotIn("val", record)

    def test_resume_matches_uninterrupted_run(self):
        full, interrupted = self.root / "full", self.root / "interrupted"
        self.run_training("final", full, "--epochs", "2")
        from bridge_seg.engine import train_one_epoch
        calls = 0

        def stop_after_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return train_one_epoch(*args, **kwargs)

        with patch("bridge_seg.engine.train_one_epoch", side_effect=stop_after_first):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_training("final", interrupted, "--epochs", "2")
        self.run_training("final", interrupted, "--epochs", "2", "--resume", str(interrupted / "last.pt"))
        expected = torch.load(full / "final.pt", map_location="cpu", weights_only=False)["model"]
        actual = torch.load(interrupted / "final.pt", map_location="cpu", weights_only=False)["model"]
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_smoke_checkpoint_cannot_resume_as_a_full_run(self):
        run_dir = self.root / "smoke"
        self.run_training("final", run_dir, "--max-train-batches", "1")
        with self.assertRaisesRegex(ValueError, "partial_run|max_train_batches"):
            self.run_training("final", run_dir, "--resume", str(run_dir / "last.pt"))

    def test_resume_rejects_changed_loss_protocol(self):
        run_dir = self.root / "protocol"
        self.run_training("final", run_dir)
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["coarse_weight"] = .7
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "coarse_weight"):
            self.run_training("final", run_dir, "--resume", str(run_dir / "last.pt"))

    def test_resume_rejects_changed_annotations(self):
        run_dir = self.root / "data-change"
        self.run_training("final", run_dir)
        source = self.prepared / "samples_clean.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["samples"][0]["semseg_ground_truth_filled"]["polylines"][0]["label"] = "ACrack"
        source.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "data_fingerprint"):
            self.run_training("final", run_dir, "--resume", str(run_dir / "last.pt"))

    def test_random_frozen_parameters_round_trip(self):
        from bridge_seg.engine import make_scaler
        torch.manual_seed(12)
        original = TinySegmentation()
        original.head.weight.requires_grad_(False)
        optimizer = torch.optim.SGD(original.parameters(), lr=.1)
        state = _checkpoint_state(original, optimizer, make_scaler(torch.device("cpu"), False),
                    1, {"pretrained": False}, None, 0, torch.Generator().manual_seed(42))
        torch.manual_seed(93)
        restored = TinySegmentation()
        restored.head.weight.requires_grad_(False)
        _load_weights(restored, state)
        image = torch.rand(1, 3, 8, 8)
        self.assertTrue(torch.equal(original(image)["logits"], restored(image)["logits"]))

    def test_changed_frozen_pretrained_weights_are_rejected(self):
        from bridge_seg.engine import make_scaler
        original = TinySegmentation()
        original.head.weight.requires_grad_(False)
        config = {"pretrained": True, "frozen_backbone_sha256": _frozen_fingerprint(original)}
        state = _checkpoint_state(original, torch.optim.SGD(original.parameters(), lr=.1),
                    make_scaler(torch.device("cpu"), False), 1, config, None, 0,
                    torch.Generator().manual_seed(42))
        restored = TinySegmentation()
        restored.head.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "Frozen pretrained backbone"):
            _load_weights(restored, state)


if __name__ == "__main__":
    unittest.main()
