from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bridge_seg.runner import infer_main

if __name__ == "__main__":
    raise SystemExit(infer_main())
