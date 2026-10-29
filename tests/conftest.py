"""Keep optional model caches and CPU test load inside this project."""
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cache = root / "outputs" / "test-hf-cache"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache))

try:
    import torch
    torch.set_num_threads(2)
except ImportError:
    pass
