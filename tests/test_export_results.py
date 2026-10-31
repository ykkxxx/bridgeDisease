import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_results.py"


class ExportTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "data/.test-tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.out = self.root / "tables"

    def make_run(self, name, seed=42, stage="dev", partial=False, epochs=2, score=.4):
        path = self.runs / name
        path.mkdir()
        config = {"model": "P", "stage": stage, "seed": seed, "epochs": epochs,
                  "image_size": 512, "partial_run": partial, "data_fingerprint": {"annotations": "abc"}}
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "metadata.json").write_text(json.dumps({"git_commit": "same-code"}), encoding="utf-8")
        rows = [{"epoch": epoch, "train": {"loss": 1 / epoch, "bce": .3, "seconds": 10},
                 "peak_memory_bytes": 1024 ** 3,
                 "val": {"miou13": score + .1 * (epoch - 1), "miou19": .3, "partial": partial,
                         "per_class": [{"label": "Crack", "iou": .5, "dice": .6, "tp": 3, "fp": 2, "fn": 1}]}}
                for epoch in (1, 2)]
        (path / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        if stage == "final":
            (path / "final.pt").write_bytes(b"fixture")
        return path

    def export(self, success=True):
        process = subprocess.run([sys.executable, str(SCRIPT), "--runs-root", str(self.runs),
                                  "--output", str(self.out)], capture_output=True, text=True)
        if success:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        else:
            self.assertNotEqual(process.returncode, 0)

    def rows(self, filename):
        with (self.out / filename).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_exports_raw_curves_and_class_counts_without_touching_history(self):
        path = self.make_run("run")
        before = (path / "history.jsonl").read_bytes()
        self.export()
        self.assertEqual((path / "history.jsonl").read_bytes(), before)
        curves = self.rows("curves.csv")
        self.assertEqual(len(curves), 2)
        self.assertEqual(float(curves[1]["val_miou13"]), .5)
        self.assertEqual(float(curves[0]["peak_memory_gib"]), 1)
        self.assertEqual(len(self.rows("per_class_metrics.csv")), 2)
        self.assertEqual(self.rows("run_summary.csv")[0]["status"], "complete")

    def test_seed_statistics_use_sample_std_and_actual_seed_count(self):
        self.make_run("s42", score=.4)
        self.make_run("s17", seed=17, score=.6)
        self.export()
        rows = self.rows("seed_summary.csv")
        row = next(row for row in rows if row["metric"] == "miou13")
        self.assertEqual(row["n_seeds"], "2")
        self.assertAlmostEqual(float(row["mean"]), .6)
        self.assertAlmostEqual(float(row["std"]), 2 ** .5 * .1)

    def test_incomplete_and_partial_runs_are_not_aggregated(self):
        self.make_run("partial", partial=True)
        self.make_run("running", seed=17, epochs=40)
        self.export()
        self.assertEqual(self.rows("seed_summary.csv"), [])
        self.assertEqual({row["status"] for row in self.rows("run_summary.csv")}, {"partial", "in_progress"})

    def test_incompatible_protocols_and_repeated_seeds_are_not_pooled(self):
        self.make_run("same42")
        self.make_run("repeat42")
        other = self.make_run("other", seed=17)
        config = json.loads((other / "config.json").read_text())
        config["image_size"] = 768
        (other / "config.json").write_text(json.dumps(config))
        self.export()
        rows = self.rows("seed_summary.csv")
        self.assertTrue(all(row["n_seeds"] == "1" for row in rows))
        report = json.loads((self.out / "export_report.json").read_text())
        self.assertTrue(any("duplicate seed" in warning for warning in report["warnings"]))

    def test_running_partial_jsonl_tail_is_ignored(self):
        path = self.make_run("run")
        with (path / "history.jsonl").open("a") as stream:
            stream.write('{"epoch":3')
        self.export()
        self.assertEqual(len(self.rows("curves.csv")), 2)

    def test_existing_output_is_not_overwritten(self):
        self.make_run("run")
        self.out.mkdir()
        self.export(success=False)

    def test_final_test_statistics_reference_the_actual_final_checkpoint(self):
        for seed, score in ((42, .5), (17, .7)):
            path = self.make_run(f"final{seed}", seed=seed, stage="final")
            report = {"split": "test", "miou13": score, "miou19": score, "partial": False,
                      "images": 966, "threshold": .5, "image_size": 512, "stride": 352,
                      "max_long_side": 1536, "manifest_sha256": "same-test",
                      "checkpoint": str(path / "final.pt"),
                      "checkpoint_sha256": hashlib.sha256(b"fixture").hexdigest(), "per_class": []}
            (path / "test_metrics.json").write_text(json.dumps(report))
        self.export()
        row = next(row for row in self.rows("seed_summary.csv") if row["metric"] == "miou13")
        self.assertEqual(row["split"], "test")
        self.assertEqual(row["n_seeds"], "2")
        self.assertAlmostEqual(float(row["mean"]), .6)

    def test_wrong_final_checkpoint_is_not_aggregated(self):
        path = self.make_run("final42", stage="final")
        report = {"split": "test", "miou13": .5, "miou19": .5, "partial": False,
                  "checkpoint": str(path / "final.pt"), "checkpoint_sha256": "wrong", "per_class": []}
        (path / "test_metrics.json").write_text(json.dumps(report))
        self.export()
        self.assertEqual(self.rows("seed_summary.csv"), [])

    def test_missing_earlier_epochs_are_not_reported_as_complete_history(self):
        path = self.make_run("gap")
        history = (path / "history.jsonl").read_text().splitlines()
        (path / "history.jsonl").write_text(history[1] + "\n")
        self.export()
        self.assertEqual(self.rows("run_summary.csv")[0]["status"], "incomplete_history")
        self.assertEqual(self.rows("seed_summary.csv"), [])


if __name__ == "__main__":
    unittest.main()
