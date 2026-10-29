"""Exercise the public packager with a small real archive and hostile inputs."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_dataset.py"
SEVENZIP = shutil.which("7z") or (
    str(Path("D:/7-Zip/7z.exe")) if Path("D:/7-Zip/7z.exe").is_file() else None
)


@unittest.skipUnless(SEVENZIP, "native 7z is required for archive integration tests")
class PackageDatasetTests(unittest.TestCase):
    def setUp(self):
        fixture_root = SCRIPT.parents[1] / "outputs" / ".package-tests"
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=fixture_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "raw"
        self.prepared = self.root / "prepared"
        self.output = self.root / "deliverables" / "clean.7z"
        (self.data / "data" / "group").mkdir(parents=True)
        for directory in ("manifests", "reports", "splits"):
            (self.prepared / directory).mkdir(parents=True)
        self.rows = []
        for index in range(5):
            relative = f"data/group/image_{index}.jpg"
            image = self.data / relative
            Image.new("RGB", (8, 6), (index * 30, 20, 90)).save(image)
            self.rows.append({
                "image_id": f"image-{index}", "filepath": relative,
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "annotation_available": index != 3,
                "annotation_index": None if index == 3 else index,
                "annotation_file": None if index == 3 else "samples_clean.json",
            })
        # An unused image and raw export must never enter the archive.
        (self.data / "samples.json").write_text('{"raw": true}', encoding="utf-8")
        (self.data / "weights.pt").write_bytes(b"not dataset content")
        (self.data / "README.md").write_text("Upstream dataset citation", encoding="utf-8")
        (self.data / "metadata.json").write_text('{"name":"fixture"}', encoding="utf-8")
        self.write_manifest("final_train", self.rows[:2])
        self.write_manifest("train", self.rows[:1])
        self.write_manifest("val", self.rows[1:2])
        self.write_manifest("test", self.rows[2:3])
        self.write_manifest("inference", self.rows[3:4])
        (self.prepared / "samples_clean.json").write_text('{"samples":[]}', encoding="utf-8")
        (self.prepared / "README.md").write_text("Prepared fixture", encoding="utf-8")
        (self.prepared / "preview.json").write_text("{}", encoding="utf-8")
        (self.prepared / "reports" / "verification.json").write_text("{}", encoding="utf-8")
        (self.prepared / "reports" / "class_distribution.csv").write_text("label,count\nCrack,3\n", encoding="utf-8")
        (self.prepared / "splits" / "train.txt").write_text("image-0\n", encoding="utf-8")
        self.write_summary()

    def write_manifest(self, name, rows):
        (self.prepared / "manifests" / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def write_summary(self):
        hashes = {
            path.relative_to(self.prepared).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.prepared.rglob("*"))
            if path.is_file() and path.name != "summary.json"
        }
        (self.prepared / "reports" / "summary.json").write_text(
            json.dumps({"artifact_sha256": hashes, "excluded_count": 1}), encoding="utf-8"
        )

    def run_cli(self, succeeds=True, extra=()):
        result = subprocess.run([
            sys.executable, str(SCRIPT), "--data-root", str(self.data),
            "--prepared-root", str(self.prepared), "--output", str(self.output),
            "--sevenzip", SEVENZIP, "--expected-image-count", "4", *extra,
        ], capture_output=True, text=True, encoding="utf-8")
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(self.output.exists())
        return result

    def test_archive_is_self_contained_deduplicated_and_hash_verified(self):
        # Catch: omitting inference, bundling raw files, or writing wrong paths/hashes.
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.data.rglob("*") if p.is_file()}
        self.run_cli()
        extraction = self.root / "extracted"
        subprocess.run([SEVENZIP, "x", str(self.output), f"-o{extraction}", "-y"],
                       check=True, capture_output=True)
        bundle = extraction / "bridge_disease_data"
        actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
        expected_images = {f"dataset/data/group/image_{i}.jpg" for i in range(4)}
        self.assertEqual({p for p in actual if p.endswith(".jpg")}, expected_images)
        expected_prepared = {f"prepared/{p.relative_to(self.prepared).as_posix()}"
                             for p in self.prepared.rglob("*") if p.is_file()}
        self.assertEqual(actual, expected_images | expected_prepared | {
            "dataset/metadata.json", "dataset/SOURCE_README.md", "PACKAGE_README.md", "package_manifest.json"
        })
        manifest = json.loads((bundle / "package_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["unique_image_count"], 4)
        self.assertEqual(manifest["split_counts"]["inference"], 1)
        self.assertEqual({entry["path"] for entry in manifest["files"]}, actual - {"package_manifest.json"})
        for entry in manifest["files"]:
            payload = (bundle / entry["path"]).read_bytes()
            self.assertEqual(entry["size"], len(payload))
            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
        sidecar = json.loads(Path(str(self.output) + ".verification.json").read_text(encoding="utf-8"))
        archive_hash = hashlib.sha256(self.output.read_bytes()).hexdigest()
        self.assertEqual(sidecar["archive_sha256"], archive_hash)
        self.assertEqual(sidecar["image_count"], 4)
        self.assertEqual(sidecar["archive_file_count"], len(actual))
        self.assertEqual(sidecar["integrity_test"], "passed")
        self.assertEqual(Path(str(self.output) + ".sha256").read_text().strip(), f"{archive_hash}  clean.7z")
        for path, (content, mtime) in before.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
        self.assertEqual(list(self.output.parent.glob("clean-staging-*")), [])

    def test_refuses_to_overwrite_an_existing_archive(self):
        # Catch: 7z updating or replacing a user archive in place.
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"keep this archive")
        result = subprocess.run([
            sys.executable, str(SCRIPT), "--data-root", str(self.data),
            "--prepared-root", str(self.prepared), "--output", str(self.output),
            "--sevenzip", SEVENZIP, "--expected-image-count", "4",
        ], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), b"keep this archive")

    def test_rejects_an_image_with_changed_content(self):
        # Catch: trusting stale manifest hashes without reading source bytes.
        (self.data / self.rows[0]["filepath"]).write_bytes(b"changed")
        result = self.run_cli(succeeds=False)
        self.assertIn("SHA256 mismatch", result.stderr)

    def test_rejects_a_manifest_path_outside_the_source(self):
        # Catch: directory traversal leaking non-dataset files.
        self.write_manifest("inference", [dict(self.rows[3], filepath="../secret.jpg")])
        self.write_summary()
        result = self.run_cli(succeeds=False)
        self.assertIn("Unsafe image path", result.stderr)

    def test_rejects_conflicting_hashes_for_the_same_image(self):
        # Catch: deduplication hiding contradictory manifest records.
        self.write_manifest("test", [dict(self.rows[0], image_sha256="0" * 64)])
        self.write_summary()
        result = self.run_cli(succeeds=False)
        self.assertIn("Conflicting SHA256", result.stderr)

    def test_rejects_changed_prepared_artifacts(self):
        # Catch: silently transporting edited annotations after cleaning verification.
        (self.prepared / "samples_clean.json").write_text('{"samples":[{}]}', encoding="utf-8")
        result = self.run_cli(succeeds=False)
        self.assertIn("SHA256 mismatch", result.stderr)

    def test_rejects_a_manifest_referring_to_unbundled_images(self):
        # Catch: leaving secondary manifests with references absent from the archive.
        self.write_manifest("train", self.rows[4:])
        self.write_summary()
        result = self.run_cli(succeeds=False)
        self.assertIn("outside the selected image union", result.stderr)

    def test_rejects_incorrect_expected_image_count(self):
        # Catch: publishing an incomplete image union as the intended dataset.
        result = self.run_cli(succeeds=False, extra=("--expected-image-count", "8851"))
        self.assertIn("Expected 8851 unique images, found 4", result.stderr)


if __name__ == "__main__":
    unittest.main()
