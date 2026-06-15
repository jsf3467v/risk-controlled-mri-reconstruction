"""Training for the cheap reconstructor: per-slice normalization, L1 plus structural loss, resumable."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "src"))

from data_processing import (M4RawSlices, slice_scale, active_device, clear_cache,
                             scan_groups, worker_count, SPLITS, SEED)
from reconstructor import Reconstructor
from metrics import ssim

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints" / "reconstructor"
EPOCHS = 30
BATCH = 4
LR = 1e-3
ACCELERATION = 4
CENTER_FRACTION = 0.08
CASCADES = 5
SSIM_WEIGHT = 0.5
KEEP_CHECKPOINTS = 3


def reconstruction_loss(model, batch, ssim_weight, device):
    """L1 plus structural loss on per-sample normalized magnitude images."""
    kspace = batch["masked_kspace"].to(device)
    mask = batch["mask"].to(device)
    target = batch["target"].to(device)
    scale = slice_scale(kspace)
    pred = model(kspace / scale.unsqueeze(1), mask)
    target = target / scale
    return F.l1_loss(pred, target) + ssim_weight * (1 - ssim(pred, target).mean())


def newest_checkpoint(folder):
    """Most recent epoch checkpoint path, or None when starting fresh."""
    points = sorted(folder.glob("epoch_*.pt"))
    return points[-1] if points else None


def prune_checkpoints(folder, keep):
    """Delete all but the most recent epoch checkpoints, leaving best.pt untouched."""
    points = sorted(folder.glob("epoch_*.pt"))
    for path in points[:-keep]:
        path.unlink()


if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = active_device()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = M4RawSlices(DATA_DIR, acceleration=ACCELERATION, center_fraction=CENTER_FRACTION)
    groups = scan_groups(dataset, SPLITS, SEED)
    workers = worker_count()
    train_loader = DataLoader(Subset(dataset, groups["train"]), batch_size=BATCH, shuffle=True,
                              num_workers=workers, persistent_workers=workers > 0)
    monitor_loader = DataLoader(Subset(dataset, groups["monitor"]), batch_size=BATCH, shuffle=False,
                                num_workers=workers, persistent_workers=workers > 0)

    model = Reconstructor(coils=4, cascades=CASCADES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch, best_val = 0, float("inf")
    latest = newest_checkpoint(CKPT_DIR)
    if latest is not None:
        state = torch.load(latest, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_val = state["epoch"] + 1, state["best_val"]
        print(f"resumed at epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = reconstruction_loss(model, batch, SSIM_WEIGHT, device)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        held = 0.0
        with torch.no_grad():
            for batch in monitor_loader:
                held += reconstruction_loss(model, batch, SSIM_WEIGHT, device).item()

        clear_cache(device)

        train_loss = running / max(len(train_loader), 1)
        val_loss = held / max(len(monitor_loader), 1)
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_val": best_val}
        torch.save(state, CKPT_DIR / f"epoch_{epoch:03d}.pt")
        if improved:
            torch.save(state, CKPT_DIR / "best.pt")
        prune_checkpoints(CKPT_DIR, KEEP_CHECKPOINTS)
        print(f"epoch {epoch:03d} | train {train_loss:.4f} | monitor {val_loss:.4f}")