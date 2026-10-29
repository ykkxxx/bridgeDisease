"""Run the monthly model/seed matrix sequentially, one GPU experiment at a time."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["B2", "B3", "B4", "P", "A2", "A4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 2026])
    parser.add_argument("--stage", choices=("dev", "final"), default="dev")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--data-root")
    parser.add_argument("--prepared-root")
    parser.add_argument("--output-root", default="outputs/month1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for model in args.models:
        config = ROOT / "configs" / "experiments" / f"{model}.json"
        if not config.is_file():
            parser.error(f"Unknown or unavailable model config: {model}")
        for seed in args.seeds:
            run_dir = Path(args.output_root) / args.stage / f"{model}_seed{seed}"
            command = [sys.executable, str(ROOT / "train.py"), "--config", str(config),
                       "--stage", args.stage, "--epochs", str(args.epochs), "--seed", str(seed),
                       "--run-dir", str(run_dir)]
            if args.data_root:
                command.extend(["--data-root", args.data_root])
            if args.prepared_root:
                command.extend(["--prepared-root", args.prepared_root])
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
