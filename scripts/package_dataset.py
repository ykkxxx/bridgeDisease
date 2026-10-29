"""Build and verify a portable cleaned dacl10k archive (Python 3.10+, native 7z).

Source data is read-only. Only selected image bytes and prepared artifacts enter
the bundle. No py7zr, Pillow, training library, or network access is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath


BUNDLE_NAME = "bridge_disease_data"
SELECTED_SPLITS = ("final_train", "test", "inference")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SIDECAR_SUFFIXES = (".sha256", ".verification.json", ".listing.txt", ".log", ".test.log")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Unsafe {description}: {value!r}")
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(value).drive or ".." in relative.parts:
        raise ValueError(f"Unsafe {description}: {value!r}")
    if not relative.parts or any("\n" in part or "\r" in part for part in relative.parts):
        raise ValueError(f"Unsafe {description}: {value!r}")
    return relative.as_posix()


def resolve_file(root: Path, relative: str, description: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Unsafe {description}: {relative!r}")
    return path


def prepared_artifacts(root: Path) -> dict[str, Path]:
    allowed_roots = {"samples_clean.json", "README.md", "preview.json"}
    allowed_suffixes = {"manifests": {".jsonl"}, "reports": {".csv", ".json"}, "splits": {".txt"}}
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe prepared artifact: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        if relative not in allowed_roots and not (
            len(parts) == 2 and path.suffix in allowed_suffixes.get(parts[0], set())
        ):
            raise ValueError(f"Unexpected prepared artifact: {relative}")
        artifacts[relative] = path
    required = {"samples_clean.json", "reports/summary.json"}
    required.update(f"manifests/{split}.jsonl" for split in SELECTED_SPLITS)
    if required - artifacts.keys():
        raise ValueError(f"Missing prepared artifacts: {sorted(required - artifacts.keys())}")
    return artifacts


def read_manifests(artifacts: dict[str, Path]) -> tuple[dict, dict, dict]:
    manifests, counts = {}, {}
    for relative, path in artifacts.items():
        if not relative.startswith("manifests/"):
            continue
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image = relative_path(row.get("filepath"), "image path")
            if PurePosixPath(image).parts[0] != "data":
                raise ValueError(f"Unsafe image path (must begin with data/): {image}")
            digest = row.get("image_sha256")
            if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
                raise ValueError(f"Missing or invalid image SHA256: {relative}:{number}")
            rows.append((image, digest))
        manifests[path.stem] = rows
        counts[path.stem] = len(rows)
    selected = {}
    for split in SELECTED_SPLITS:
        for image, digest in manifests[split]:
            if image in selected and selected[image] != digest:
                raise ValueError(f"Conflicting SHA256 for image: {image}")
            selected[image] = digest
    for split, rows in manifests.items():
        for image, digest in rows:
            if image not in selected:
                raise ValueError(f"Manifest {split} references image outside the selected image union: {image}")
            if selected[image] != digest:
                raise ValueError(f"Conflicting SHA256 for image: {image}")
    return selected, manifests, counts


def verify_prepared(artifacts: dict[str, Path]) -> dict[str, str]:
    actual = {relative: sha256_file(path) for relative, path in artifacts.items()}
    summary = json.loads(artifacts["reports/summary.json"].read_text(encoding="utf-8"))
    expected = summary.get("artifact_sha256")
    if not isinstance(expected, dict) or "samples_clean.json" not in expected:
        raise ValueError("Prepared summary must contain artifact_sha256 including samples_clean.json")
    for value, digest in expected.items():
        relative = relative_path(value, "prepared path")
        if relative not in actual:
            raise ValueError(f"Prepared summary references missing artifact: {relative}")
        if actual[relative] != digest:
            raise ValueError(f"SHA256 mismatch for prepared artifact: {relative}")
    return actual


def readme(split_counts: dict[str, int], unique_count: int) -> str:
    return f"""# Cleaned dacl10k data for AutoDL

This bundle contains {unique_count} unique original images, cleaned polygon
annotations and all prepared manifests, split IDs and cleaning reports.
Image bytes and original polygon coordinates have not been changed.

## Paths after extraction

Extract this archive into /root/autodl-tmp. Configure training with:

    data_root: /root/autodl-tmp/bridge_disease_data/dataset
    prepared_root: /root/autodl-tmp/bridge_disease_data/prepared

Each JSONL filepath is relative to data_root. annotation_index addresses the
samples array in prepared/samples_clean.json. No raw samples.json is needed.
Historical Windows paths inside audit reports describe the source machine;
they are provenance, not paths to use on AutoDL.

## Split protocol

{chr(10).join(f'- {name}: {count}' for name, count in sorted(split_counts.items()))}

train / val form the internal training and tuning split. final_train is their
union, used only after locking the protocol. test (alias final_eval) is the
cleaned nonempty official validation subset, reserved for final evaluation.
It is not the complete original 975-image official validation benchmark.
inference contains testdev images with NO local ground truth: produce predictions
only; do not report local ground-truth segmentation scores for this split.
train_inner / tune_inner are aliases for train / val, not extra images.

## Verification and provenance

Before extracting, run `7z t dacl10k_clean_autodl.7z` and compare the archive's
SHA256 to the accompanying .sha256 file. package_manifest.json records the
relative path, byte size and SHA256 of every other file in this bundle. The
manifest excludes itself to avoid a circular hash. Original image hashes were
fully recomputed and checked against the prepared manifests before packaging.

dataset/SOURCE_README.md and dataset/metadata.json, when present, are copied
upstream attribution and metadata. Their original licensing statements are
preserved. This is a training bundle, not a complete FiftyOne export.
See prepared/reports/summary.json and the other reports for cleaning choices,
exclusions, coordinate warnings and limitations. Overlapping classes require
independent binary masks; polygon boundaries require geometric clipping.
Raw samples.json, excluded images, model weights and caches are not included.
"""


def run_7z(command: list[str], cwd: Path, log: Path) -> None:
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        detail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"7z exited with code {result.returncode}:\n{detail}")


def listed_files(text: str) -> dict[str, int]:
    files = {}
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        record = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if "Path" not in record:
            continue
        if record.get("Folder") == "+" or record.get("Attributes", "").lstrip().startswith("D"):
            continue
        path = record["Path"].replace("\\", "/")
        if path in files:
            raise ValueError(f"Duplicate archive member: {path}")
        files[path] = int(record["Size"])
    return files


def publish_exclusive(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)


def create_package(args: argparse.Namespace) -> dict:
    data_root = args.data_root.resolve(strict=True)
    prepared_root = args.prepared_root.resolve(strict=True)
    output = args.output.resolve()
    if not data_root.is_dir() or not prepared_root.is_dir():
        raise ValueError("data-root and prepared-root must be directories")
    if output.suffix.lower() != ".7z":
        raise ValueError("Output must use the .7z extension")
    if output.is_relative_to(data_root) or output.is_relative_to(prepared_root):
        raise ValueError("Output must be outside both read-only input roots")
    for path in [output, *(Path(str(output) + suffix) for suffix in SIDECAR_SUFFIXES)]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    sevenzip = shutil.which(str(args.sevenzip)) or str(args.sevenzip)
    if not Path(sevenzip).is_file():
        raise FileNotFoundError(f"Native 7z executable not found: {args.sevenzip}")
    artifacts = prepared_artifacts(prepared_root)
    prepared_hashes = verify_prepared(artifacts)
    selected, _manifests, split_counts = read_manifests(artifacts)
    if len(selected) != args.expected_image_count:
        raise ValueError(f"Expected {args.expected_image_count} unique images, found {len(selected)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{output.stem}-staging-", dir=output.parent) as staging_dir:
        staging = Path(staging_dir)
        bundle = staging / BUNDLE_NAME
        bundle.mkdir()
        entries = []
        hardlinked_images = 0
        print(f"Staging and fully verifying {len(selected)} images: {staging}", flush=True)
        for index, (relative, expected_hash) in enumerate(sorted(selected.items()), 1):
            source = resolve_file(data_root, relative, "image path")
            actual_hash = sha256_file(source)
            if actual_hash != expected_hash:
                raise ValueError(f"SHA256 mismatch for source image: {relative}")
            destination = bundle / "dataset" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
                hardlinked_images += 1
            except OSError:
                shutil.copyfile(source, destination)
                if sha256_file(destination) != actual_hash:
                    raise ValueError(f"SHA256 mismatch for staged image: {relative}")
            entries.append({"path": f"dataset/{relative}", "size": destination.stat().st_size, "sha256": actual_hash})
            if index % 500 == 0 or index == len(selected):
                print(f"Verified and staged {index}/{len(selected)} images", flush=True)
        for relative, source in artifacts.items():
            destination = bundle / "prepared" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            digest = sha256_file(destination)
            if digest != prepared_hashes[relative]:
                raise ValueError(f"SHA256 mismatch for staged prepared artifact: {relative}")
            entries.append({"path": f"prepared/{relative}", "size": destination.stat().st_size, "sha256": digest})
        for original, packaged in (("README.md", "SOURCE_README.md"), ("metadata.json", "metadata.json")):
            if (data_root / original).is_file():
                source = resolve_file(data_root, original, "metadata path")
                destination = bundle / "dataset" / packaged
                shutil.copyfile(source, destination)
                entries.append({"path": f"dataset/{packaged}", "size": destination.stat().st_size,
                                "sha256": sha256_file(destination)})
        bundle_readme = bundle / "PACKAGE_README.md"
        bundle_readme.write_text(readme(split_counts, len(selected)), encoding="utf-8", newline="\n")
        entries.append({"path": "PACKAGE_README.md", "size": bundle_readme.stat().st_size,
                        "sha256": sha256_file(bundle_readme)})
        manifest = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": "Voxel51/dacl10k", "selected_splits": list(SELECTED_SPLITS),
            "unique_image_count": len(selected), "split_counts": split_counts,
            "image_verification": "all source image SHA256 recomputed and matched to manifests",
            "manifest_excludes_itself": True,
            "cloud_paths": {"data_root": f"/root/autodl-tmp/{BUNDLE_NAME}/dataset",
                            "prepared_root": f"/root/autodl-tmp/{BUNDLE_NAME}/prepared"},
            "files": sorted(entries, key=lambda entry: entry["path"]),
        }
        manifest_path = bundle / "package_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        staged_archive = staging / "payload.7z"
        compression_log, test_log, listing_log = (staging / name for name in ("compression.log", "test.log", "listing.txt"))
        print(f"Compressing with native 7z (log: {compression_log})", flush=True)
        run_7z([sevenzip, "a", "-t7z", f"-mx={args.compression}", f"-ms={args.solid}",
                "-mmt=on", "-bsp0", "-bb0", "-sccUTF-8", str(staged_archive), BUNDLE_NAME], staging, compression_log)
        print("Testing the complete archive with 7z t", flush=True)
        run_7z([sevenzip, "t", "-bsp0", "-sccUTF-8", str(staged_archive)], staging, test_log)
        run_7z([sevenzip, "l", "-slt", "-ba", "-sccUTF-8", str(staged_archive)], staging, listing_log)
        actual_files = listed_files(listing_log.read_text(encoding="utf-8-sig"))
        expected_files = {f"{BUNDLE_NAME}/{entry['path']}": entry["size"] for entry in entries}
        expected_files[f"{BUNDLE_NAME}/package_manifest.json"] = manifest_path.stat().st_size
        if actual_files != expected_files:
            raise ValueError("Archive listing does not exactly match the staged file paths and sizes")
        print("Computing the external archive SHA256", flush=True)
        archive_hash = sha256_file(staged_archive)
        report = {
            "archive": str(output), "archive_size": staged_archive.stat().st_size,
            "archive_sha256": archive_hash, "integrity_test": "passed",
            "listing_matches_manifest": True, "archive_file_count": len(actual_files),
            "image_count": len(selected), "prepared_file_count": len(artifacts),
            "split_counts": split_counts, "hardlinked_images": hardlinked_images,
            "source_image_hashes_verified": len(selected),
            "payload_bytes": sum(actual_files.values()),
        }
        publish_exclusive(staged_archive, output)
        for suffix, content in (
            (".sha256", f"{archive_hash}  {output.name}\n"),
            (".verification.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
        ):
            with Path(str(output) + suffix).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
        for source, suffix in ((compression_log, ".log"), (test_log, ".test.log"), (listing_log, ".listing.txt")):
            publish_exclusive(source, Path(str(output) + suffix))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/dataset/dacl10k"))
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dacl10k_clean_autodl.7z"))
    default_sevenzip = shutil.which("7z") or ("D:/7-Zip/7z.exe" if Path("D:/7-Zip/7z.exe").is_file() else "7z")
    parser.add_argument("--sevenzip", type=Path, default=Path(default_sevenzip))
    parser.add_argument("--expected-image-count", type=int, default=8851)
    parser.add_argument("--compression", type=int, choices=range(10), default=1)
    parser.add_argument("--solid", choices=("on", "off"), default="off")
    args = parser.parse_args()
    try:
        report = create_package(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
