"""Calibrate the per-contrast (Mondrian) gate. Fit one risk-controlled threshold per contrast on the held-out calibrate split."""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "src"))

from data_processing import (M4RawSlices, active_device, scan_groups,
                             worker_count, SPLITS, SEED)
from reconstructor import Reconstructor
from metrics import slice_scores
from gate import mondrian_thresholds, accepted_groups

DATA_DIR = ROOT / "data"
CHECKPOINT = ROOT / "checkpoints" / "reconstructor" / "best.pt"
GATE_PATH = ROOT / "checkpoints" / "gate.json"
BATCH = 4
ACCELERATION = 4
CENTER_FRACTION = 0.08
CASCADES = 5
EPSILON = 0.20
ALPHA = 0.1
MINIMUM = 50
CANDIDATES = 8


if __name__ == "__main__":
    device = active_device()
    dataset = M4RawSlices(DATA_DIR, acceleration=ACCELERATION, center_fraction=CENTER_FRACTION)
    groups = scan_groups(dataset, SPLITS, SEED)
    workers = worker_count()
    loader = DataLoader(Subset(dataset, groups["calibrate"]), batch_size=BATCH, shuffle=False,
                        num_workers=workers, persistent_workers=workers > 0)
    model = Reconstructor(coils=4, cascades=CASCADES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["model"])
    model.eval()

    scores, errors, contrasts = slice_scores(model, loader, device)
    thresholds = mondrian_thresholds(scores, errors, contrasts, EPSILON, ALPHA, MINIMUM, CANDIDATES)
    keep = accepted_groups(scores, contrasts, thresholds)

    for name in sorted(thresholds):
        pick = contrasts == name
        rate = float(keep[pick].mean()) if pick.any() else 0.0
        print(f"{name:5s} n={int(pick.sum()):4d} | threshold {thresholds[name]:.4f} | accept {rate:5.1%}")
    overall = float(keep.mean())
    print(f"overall | accept {overall:.1%} | escalate {1 - overall:.1%}")

    GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_PATH.write_text(json.dumps(
        {"epsilon": EPSILON, "alpha": ALPHA, "acceleration": ACCELERATION,
         "minimum": MINIMUM, "candidates": CANDIDATES, "thresholds": thresholds,
         "accept_rate": overall}, indent=2))