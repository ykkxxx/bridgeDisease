"""Small real-image fixtures for geometry, masking, and accumulated metrics."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bridge_seg.data import BridgeDataset
from bridge_seg.losses import multilabel_loss, segmentation_loss
from bridge_seg.metrics import MultiLabelMetrics


class DatasetTests(unittest.TestCase):
    def setUp(self):
        fixture_root = Path(__file__).resolve().parents[1] / "data" / ".test-tmp"
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=fixture_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "dataset"
        (self.data / "data").mkdir(parents=True)
        self.prepared = self.root / "prepared"
        (self.prepared / "manifests").mkdir(parents=True)

    def fixture(self, regions, size=(12, 8), pixels=None, available=True, exif=None):
        image = Image.new("RGB", size, (255, 0, 0)) if pixels is None else Image.fromarray(pixels)
        relative = "data/fixture.jpg" if exif else "data/fixture.png"
        image.save(self.data / relative, **({"exif": exif} if exif else {}))
        # Incorrect export metadata must never replace the actual decoded size.
        sample = {"_id": {"$oid": "fixture"}, "filepath": relative,
                  "metadata": {"width": 999, "height": 777},
                  "semseg_ground_truth_filled": {"polylines": regions}}
        (self.prepared / "samples_clean.json").write_text(json.dumps({"samples": [sample]}))
        row = {"image_id": "fixture", "filepath": relative, "annotation_index": 0,
               "annotation_file": "samples_clean.json", "annotation_available": available,
               "width": 999, "height": 777}
        (self.prepared / "manifests" / "train.jsonl").write_text(json.dumps(row) + "\n")
        return row

    @staticmethod
    def region(label, points):
        return {"label": label, "points": [points], "closed": True, "filled": True}

    def test_original_geometry_and_cross_class_overlap_survive_rasterization(self):
        self.fixture([
            self.region("Crack", [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]),
            self.region("Rust", [[.4, .3], [.8, .3], [.8, .8], [.4, .8]]),
        ])
        sample = BridgeDataset(self.data, self.prepared, image_size=None)[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 8, 12))
        self.assertEqual(tuple(sample["target"].shape), (19, 8, 12))
        self.assertEqual(sample["valid"].dtype, torch.bool)
        self.assertTrue(sample["valid"].all())
        self.assertEqual(sample["target"][0, 4, 6].item(), 1)
        self.assertEqual(sample["target"][4, 4, 6].item(), 1)
        self.assertEqual(sample["target"][1].sum().item(), 0)
        self.assertEqual(sample["image"].max().item(), 1)
        self.assertEqual(sample["image"].min().item(), 0)
        self.assertEqual(sample["image_id"], "fixture")

    def test_outside_triangle_is_geometrically_clipped_not_vertex_clamped(self):
        self.fixture([self.region("Crack", [[-1, 0], [.5, .5], [-1, 1]])], size=(12, 12))
        target = BridgeDataset(self.data, self.prepared, image_size=None)[0]["target"][0]
        # True left-border intersections are y=4 and y=8. Vertex clamping would
        # create a false full-height triangle along the left border.
        self.assertEqual(target[1, 0].item(), 0)
        self.assertEqual(target[10, 0].item(), 0)
        self.assertEqual(target[6, 0].item(), 1)
        self.assertEqual(target[6, 4].item(), 1)
        self.assertEqual(target[6, 9].item(), 0)

    def test_eval_resize_preserves_aspect_and_marks_padding_invalid(self):
        self.fixture([self.region("Crack", [[0, 0], [1, 0], [1, 1], [0, 1]])], size=(8, 4))
        sample = BridgeDataset(self.data, self.prepared, image_size=8)[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 8, 8))
        self.assertEqual(sample["valid"].sum().item(), 32)
        self.assertTrue(sample["valid"][0, 2:6].all())
        self.assertFalse(sample["valid"][0, :2].any())
        self.assertTrue(torch.equal(sample["target"][0].bool(), sample["valid"][0]))

    def test_random_crop_and_flips_keep_pixels_and_targets_aligned(self):
        pixels = np.zeros((4, 8, 3), dtype=np.uint8)
        pixels[:, :5, 0] = 255
        self.fixture([self.region("Crack", [[0, 0], [.5, 0], [.5, 1], [0, 1]])],
                     size=(8, 4), pixels=pixels)
        dataset = BridgeDataset(self.data, self.prepared, image_size=4, training=True,
                                max_long_side=8, scale_range=(1., 1.))
        for seed in range(8):
            torch.manual_seed(seed)
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 4, 4))
            self.assertTrue(torch.equal(sample["image"][0] > .5, sample["target"][0].bool()))
            self.assertTrue(sample["valid"].all())

    def test_training_padding_is_invalid_and_contains_no_target(self):
        self.fixture([self.region("Crack", [[0, 0], [1, 0], [1, 1], [0, 1]])], size=(8, 4))
        torch.manual_seed(5)
        sample = BridgeDataset(self.data, self.prepared, image_size=8, training=True,
                               max_long_side=8, scale_range=(1., 1.))[0]
        self.assertEqual(sample["valid"].sum().item(), 32)
        self.assertTrue(torch.equal(sample["image"][0] > .5, sample["valid"][0]))
        self.assertTrue(torch.equal(sample["target"][0].bool(), sample["valid"][0]))

    def test_training_random_scale_changes_geometry_within_declared_range(self):
        self.fixture([], size=(16, 8))
        dataset = BridgeDataset(self.data, self.prepared, image_size=32, training=True, max_long_side=16)
        valid_areas = set()
        for seed in range(8):
            torch.manual_seed(seed)
            valid = dataset[0]["valid"][0]
            rows, cols = torch.where(valid)
            width = cols.max().item() - cols.min().item() + 1
            height = rows.max().item() - rows.min().item() + 1
            self.assertGreaterEqual(width, 12)
            self.assertLessEqual(width, 20)
            self.assertLessEqual(abs(width / height - 2), .2)
            valid_areas.add(valid.sum().item())
        self.assertGreater(len(valid_areas), 1)

    def test_exif_orientation_does_not_rotate_only_the_image(self):
        exif = Image.Exif()
        exif[274] = 6
        self.fixture([self.region("Crack", [[0, 0], [1, 0], [1, 1], [0, 1]])],
                     size=(8, 4), exif=exif)
        sample = BridgeDataset(self.data, self.prepared, image_size=None)[0]
        self.assertEqual(tuple(sample["image"].shape[-2:]), (4, 8))
        self.assertEqual(tuple(sample["target"].shape[-2:]), (4, 8))

    def test_unlabeled_records_cannot_be_scored_as_negative_targets(self):
        self.fixture([], available=False)
        with self.assertRaises(ValueError):
            BridgeDataset(self.data, self.prepared)

    def test_prepared_index_must_match_the_manifest_filepath(self):
        row = self.fixture([])
        row["filepath"] = "data/different.png"
        (self.prepared / "manifests" / "train.jsonl").write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            BridgeDataset(self.data, self.prepared)


class LossTests(unittest.TestCase):
    def test_all_negative_dice_is_zero_but_bce_still_trains(self):
        logits = torch.zeros(2, 19, 2, 2, requires_grad=True)
        target = torch.zeros_like(logits)
        valid = torch.ones(2, 1, 2, 2, dtype=torch.bool)
        loss, parts = segmentation_loss({"logits": logits}, target, valid)
        self.assertAlmostEqual(parts["dice"].item(), 0)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertAlmostEqual(multilabel_loss(logits, target, valid).item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue((logits.grad > 0).all())

    def test_padding_has_no_effect_on_main_loss_or_gradients(self):
        logits = torch.tensor([[[[0., 100.], [0., -100.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 0.], [1., 1.]]]])
        valid = torch.tensor([[[[True, False], [True, False]]]])
        actual = multilabel_loss(logits, target, valid)
        # Two valid positive pixels, each p=.5: BCE=log2, Dice loss=1/3.
        self.assertAlmostEqual(actual.item(), math.log(2) + 1/3, places=5)
        actual.backward()
        self.assertEqual(logits.grad[0, 0, 0, 1].item(), 0)
        self.assertEqual(logits.grad[0, 0, 1, 1].item(), 0)

    def test_dice_averages_positive_classes_per_image_not_across_batch(self):
        logits = torch.zeros(2, 2, 1, 2)
        target = torch.tensor([[[[1., 1.]], [[0., 0.]]], [[[1., 0.]], [[1., 0.]]]])
        valid = torch.ones(2, 1, 1, 2, dtype=torch.bool)
        # Image 0: loss1/3; image1: each positive class loss1/2.
        expected_dice = (1/3 + .5) / 2
        self.assertAlmostEqual(multilabel_loss(logits, target, valid).item(),
                               math.log(2) + expected_dice, places=5)

    def test_coarse_loss_is_average_bce_and_optional_edge_is_differentiable(self):
        target = torch.zeros(1, 2, 7, 7)
        target[:, 0, 2:5, 2:5] = 1
        valid = torch.ones(1, 1, 7, 7, dtype=torch.bool)
        logits = torch.zeros_like(target, requires_grad=True)
        coarse = [torch.zeros(1, 2, 2, 2, requires_grad=True) for _ in range(4)]
        edge = torch.zeros_like(target, requires_grad=True)
        loss, parts = segmentation_loss({"logits": logits, "coarse": coarse, "edge": edge}, target, valid)
        self.assertAlmostEqual(parts["coarse"].item(), math.log(2), places=6)
        self.assertAlmostEqual(loss.item(), parts["main"].item() + .2 * parts["coarse"].item()
                               + .1 * parts["edge"].item(), places=6)
        loss.backward()
        self.assertGreater(edge.grad.abs().sum().item(), 0)
        self.assertTrue(all(c.grad is not None and c.grad.abs().sum() > 0 for c in coarse))

    def test_edge_loss_erodes_padding_neighbors(self):
        target = torch.zeros(1, 1, 7, 7)
        target[:, :, 1:5, 1:4] = 1
        valid = torch.ones(1, 1, 7, 7, dtype=torch.bool)
        valid[:, :, :, 4:] = False
        first = torch.zeros_like(target)
        second = first.clone()
        # x=3 is valid segmentation territory but touches invalid padding.
        second[:, :, :, 3:] = 100
        a, a_parts = segmentation_loss({"logits": first, "edge": first}, target, valid)
        b, b_parts = segmentation_loss({"logits": first, "edge": second}, target, valid)
        self.assertAlmostEqual(a_parts["edge"].item(), b_parts["edge"].item(), places=6)
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_all_invalid_batch_has_finite_zero_loss_and_zero_gradient(self):
        logits = torch.randn(1, 19, 5, 5, requires_grad=True)
        target = torch.zeros_like(logits)
        valid = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
        loss, _ = segmentation_loss({"logits": logits, "coarse": [logits], "edge": logits}, target, valid)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)


class MetricTests(unittest.TestCase):
    def test_confusion_counts_accumulate_before_iou_and_ignore_invalid_pixels(self):
        metric = MultiLabelMetrics(num_classes=2)
        logits = torch.tensor([[[[2., 2., -2., -2.]], [[-2., -2., -2., 2.]]]])
        target = torch.tensor([[[[1., 0., 1., 0.]], [[0., 0., 0., 1.]]]])
        valid = torch.tensor([[[[True, True, True, False]]]])
        metric.update(logits, target, valid)
        # Second image adds four true positives to class0. Aggregate IoU=5/7;
        # averaging the two image IoUs would incorrectly produce2/3.
        metric.update(torch.tensor([[[[2., 2., 2., 2.]], [[-2., -2., -2., -2.]]]]),
                      torch.tensor([[[[1., 1., 1., 1.]], [[0., 0., 0., 0.]]]]),
                      torch.ones(1, 1, 1, 4, dtype=torch.bool))
        result = metric.compute()
        cls = result["per_class"][0]
        self.assertEqual((cls["tp"], cls["fp"], cls["fn"]), (5, 1, 1))
        self.assertAlmostEqual(cls["iou"], 5/7)
        self.assertAlmostEqual(cls["dice"], 10/12)
        self.assertAlmostEqual(cls["precision"], 5/6)
        self.assertAlmostEqual(cls["recall"], 5/6)
        self.assertIsNone(result["per_class"][1]["iou"])
        self.assertAlmostEqual(result["miou19"], 5/7)
        json.dumps(result, allow_nan=False)

    def test_inputs_are_logits_and_undefined_results_are_json_null(self):
        metric = MultiLabelMetrics(num_classes=19)
        logits = torch.full((1, 19, 1, 1), -.1)
        metric.update(logits, torch.zeros_like(logits), torch.ones(1, 1, 1, 1, dtype=torch.bool))
        result = metric.compute()
        self.assertIsNone(result["miou13"])
        self.assertIsNone(result["miou19"])
        self.assertTrue(all(row["fp"] == 0 and row["iou"] is None for row in result["per_class"]))
        self.assertIn("null", json.dumps(result, allow_nan=False))

    def test_miou13_uses_only_first_thirteen_disease_classes(self):
        metric = MultiLabelMetrics(num_classes=19)
        logits = torch.full((1, 19, 1, 1), -2.)
        target = torch.zeros_like(logits)
        logits[:, 0] = 2
        target[:, 0] = 1
        logits[:, 13] = 2  # Structure class with one false positive.
        metric.update(logits, target, torch.ones(1, 1, 1, 1, dtype=torch.bool))
        result = metric.compute()
        self.assertEqual(result["miou13"], 1)
        self.assertEqual(result["miou19"], .5)


if __name__ == "__main__":
    unittest.main()
