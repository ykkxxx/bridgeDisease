"""Integration tests use real images and the public CLI, not mocked I/O."""
import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"


class PreparationTests(unittest.TestCase):
    def setUp(self):
        fixture_root = SCRIPT.parents[1] / "data" / ".test-tmp"
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=fixture_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "raw"
        (self.data / "data").mkdir(parents=True)
        self.output = self.root / "prepared"
        self.samples = []
        for split, count in (("train", 20), ("validation", 4), ("testdev", 2)):
            for _ in range(count):
                self.add_sample(split)
        self.save_source()

    def add_sample(self, split, empty=False):
        index = len(self.samples)
        name = f"data/image_{index:03d}.png"
        Image.new("RGB", (12, 8), (index, index * 3, index * 7)).save(self.data / name)
        region = {
            "label": "Crack", "closed": True, "filled": True,
            "points": [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]],
        }
        row = {
            "_id": {"$oid": f"id-{index}"}, "filepath": name, "tags": [split],
            "metadata": None if split == "testdev" else {"width": 12, "height": 8},
            "semseg_ground_truth_filled": None if split == "testdev" else {
                "polylines": [] if empty else [region]
            },
        }
        self.samples.append(row)
        return row

    def save_source(self):
        path = self.data / "samples.json"
        path.write_text(json.dumps({"samples": self.samples}), encoding="utf-8")
        return path

    def run_cli(self, *args, succeeds=True, output=None):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--data-root", str(self.data),
             "--output", str(output or self.output), *args],
            text=True, capture_output=True, encoding="utf-8",
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def rows(self, split, output=None):
        path = (output or self.output) / "manifests" / f"{split}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def summary(self):
        return json.loads((self.output / "reports" / "summary.json").read_text(encoding="utf-8"))

    def test_research_splits_are_disjoint_and_preserve_source(self):
        source = self.save_source()
        before = source.read_bytes()
        self.run_cli()
        self.assertEqual(before, source.read_bytes())
        expected = {"train": 18, "val": 2, "test": 4, "inference": 2, "final_train": 20}
        self.assertEqual(self.summary()["counts"], expected)
        groups = {name: {r["filepath"] for r in self.rows(name)} for name in expected}
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["final_train"] & groups["test"])
        self.assertEqual(groups["train"] | groups["val"], groups["final_train"])
        self.assertTrue(all(r["source_split"] == "validation" for r in self.rows("test")))
        self.assertTrue(all(not r["annotation_available"] for r in self.rows("inference")))
        self.assertTrue(all((r["width"], r["height"]) == (12, 8) for r in self.rows("inference")))
        self.assertEqual(len(self.summary()["class_order"]), 19)
        self.assertEqual(self.summary()["source_sha256"], hashlib.sha256(before).hexdigest())
        clean = json.loads((self.output / "samples_clean.json").read_text(encoding="utf-8"))["samples"]
        for split in ("train", "val", "test", "final_train"):
            for row in self.rows(split):
                self.assertEqual(clean[row["annotation_index"]]["filepath"], row["filepath"])
        for first, alias in (("train", "train_inner"), ("val", "tune_inner"), ("test", "final_eval")):
            self.assertEqual(self.rows(first), self.rows(alias))

    def test_empty_annotations_are_excluded_but_inference_is_preserved(self):
        self.add_sample("train", empty=True)
        self.add_sample("validation", empty=True)
        self.save_source()
        self.run_cli()
        self.assertEqual(self.summary()["excluded_reasons"]["empty_segmentation_regions"], 2)
        self.assertEqual(len(self.rows("inference")), 2)
        self.assertEqual(len(self.rows("test")), 4)

    def test_empty_annotations_can_be_explicitly_kept(self):
        self.add_sample("train", empty=True)
        self.add_sample("validation", empty=True)
        self.save_source()
        self.run_cli("--empty-policy", "keep")
        self.assertEqual(len(self.rows("final_train")), 21)
        self.assertEqual(len(self.rows("test")), 5)
        self.assertIn([], [r["labels"] for r in self.rows("test")])

    def test_missing_labels_are_never_zero_negative_samples(self):
        self.samples[0]["semseg_ground_truth_filled"] = None
        self.save_source()
        self.run_cli("--empty-policy", "keep")
        self.assertEqual(self.summary()["excluded_reasons"]["missing_segmentation_annotation"], 1)
        self.assertNotIn(self.samples[0]["filepath"], {r["filepath"] for r in self.rows("final_train")})

    def test_identical_images_stay_in_one_inner_subset(self):
        duplicate = self.add_sample("train")
        shutil.copyfile(self.data / self.samples[0]["filepath"], self.data / duplicate["filepath"])
        self.save_source()
        self.run_cli()
        train = {r["filepath"] for r in self.rows("train")}
        val = {r["filepath"] for r in self.rows("val")}
        duplicate_names = {duplicate["filepath"], self.samples[0]["filepath"]}
        self.assertTrue(duplicate_names <= train or duplicate_names <= val)
        self.assertTrue((self.output / "reports" / "duplicate_groups.csv").exists())

    def test_training_copy_of_final_test_is_excluded(self):
        shutil.copyfile(self.data / self.samples[0]["filepath"], self.data / self.samples[20]["filepath"])
        self.run_cli()
        self.assertEqual(len(self.rows("test")), 4)
        self.assertEqual(len(self.rows("final_train")), 19)
        self.assertEqual(self.summary()["excluded_reasons"]["duplicates_final_test"], 1)
        self.assertFalse({r["image_sha256"] for r in self.rows("final_train")} &
                         {r["image_sha256"] for r in self.rows("test")})

    def test_final_test_wins_cross_source_path_and_id_collisions(self):
        self.samples[20]["filepath"] = self.samples[0]["filepath"]
        self.samples[21]["_id"] = copy.deepcopy(self.samples[1]["_id"])
        # Another train copy must also be excluded after the path collision.
        shutil.copyfile(self.data / self.samples[0]["filepath"], self.data / self.samples[2]["filepath"])
        self.save_source()
        self.run_cli()
        self.assertEqual(len(self.rows("test")), 4)
        self.assertTrue({r["source_split"] for r in self.rows("test")} == {"validation"})
        self.assertFalse({r["image_sha256"] for r in self.rows("final_train")} &
                         {r["image_sha256"] for r in self.rows("test")})

    def test_large_duplicate_group_does_not_force_ratio_overshoot(self):
        original = self.data / self.samples[0]["filepath"]
        for sample in self.samples[1:8]:
            shutil.copyfile(original, self.data / sample["filepath"])
        hashes = {hashlib.sha256((self.data / row["filepath"]).read_bytes()).hexdigest()
                  for row in self.samples[:20]}
        large_group = hashlib.sha256(original.read_bytes()).hexdigest()
        seed = next(value for value in range(1000) if min(
            hashes, key=lambda key: hashlib.sha256(f"{value}:{key}".encode("ascii")).hexdigest()
        ) == large_group)
        self.run_cli("--seed", str(seed))
        self.assertEqual(len(self.rows("val")), 2)

    def test_seed_is_reproducible_even_if_source_order_changes(self):
        self.run_cli()
        expected = {r["filepath"] for r in self.rows("val")}
        random.Random(19).shuffle(self.samples)
        self.save_source()
        self.run_cli(output=self.root / "reordered")
        self.assertEqual(expected, {r["filepath"] for r in self.rows("val", self.root / "reordered")})
        self.run_cli("--seed", "17", output=self.root / "other_seed")
        self.assertNotEqual(expected, {r["filepath"] for r in self.rows("val", self.root / "other_seed")})

    def test_validation_ratio_is_configurable_and_invalid_values_are_rejected(self):
        self.run_cli("--val-ratio", "0.2")
        self.assertEqual(len(self.rows("val")), 4)
        for index, value in enumerate(("0", "1", "-0.1", "nan")):
            with self.subTest(value=value):
                dest = self.root / f"invalid-{index}"
                self.run_cli("--val-ratio", value, output=dest, succeeds=False)
                self.assertFalse(dest.exists())

    def test_corrupt_and_missing_images_are_reported(self):
        (self.data / self.samples[0]["filepath"]).write_bytes(b"not an image")
        (self.data / self.samples[1]["filepath"]).unlink()
        self.run_cli()
        self.assertEqual(self.summary()["excluded_reasons"]["unreadable_image"], 1)
        self.assertEqual(self.summary()["excluded_reasons"]["missing_image"], 1)

    def test_unknown_labels_and_nonfinite_coordinates_are_reported(self):
        self.samples[0]["semseg_ground_truth_filled"]["polylines"][0]["label"] = "UnknownClass"
        self.samples[1]["semseg_ground_truth_filled"]["polylines"][0]["points"][0][0][0] = float("nan")
        self.save_source()
        self.run_cli()
        self.assertEqual(self.summary()["excluded_reasons"]["unknown_label"], 1)
        self.assertEqual(self.summary()["excluded_reasons"]["invalid_polygon"], 1)

    def test_out_of_bounds_coordinates_are_preserved(self):
        region = self.samples[0]["semseg_ground_truth_filled"]["polylines"][0]
        region["points"][0][0][0] = -0.05
        expected = copy.deepcopy(self.samples[0])
        self.save_source()
        self.run_cli()
        self.assertEqual(self.summary()["coordinate_audit"]["out_of_bounds_vertices"], 1)
        clean = json.loads((self.output / "samples_clean.json").read_text(encoding="utf-8"))["samples"]
        self.assertIn(expected, clean)

    def test_unsafe_paths_are_excluded(self):
        self.samples[0]["filepath"] = "../outside.png"
        self.samples[1]["filepath"] = "C:\\outside.png"
        self.save_source()
        self.run_cli()
        self.assertEqual(self.summary()["excluded_reasons"]["unsafe_image_path"], 2)

    def test_duplicate_paths_and_ids_are_excluded(self):
        self.samples[1]["filepath"] = self.samples[0]["filepath"]
        self.samples[3]["_id"] = copy.deepcopy(self.samples[2]["_id"])
        self.save_source()
        self.run_cli()
        self.assertEqual(self.summary()["excluded_reasons"]["duplicate_filepath"], 1)
        self.assertEqual(self.summary()["excluded_reasons"]["duplicate_image_id"], 1)

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "marker.txt"
        marker.write_text("preserve", encoding="utf-8")
        self.run_cli(succeeds=False)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_output_inside_raw_dataset_is_rejected(self):
        dest = self.data / "prepared"
        self.run_cli(output=dest, succeeds=False)
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
