"""Evaluation of the certified reconstruction system on the eval split. All in one pass, the reconstructor quality
against the zero-filled baseline, the held-out gate's realized risk and per-contrast certify or
flag rates, and a per-contrast qualitative panel. The README describes the
report and the figure. Accepted slices are certified at risk at or under epsilon, the rest are
flagged for review.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from data_processing import (M4RawSlices, zero_filled, slice_scale, active_device,
                             clear_cache, scan_groups, worker_count, SPLITS, SEED)
from reconstructor import Reconstructor
from metrics import ssim, psnr, nmse, contrast_auroc, contrast_aurc
from gate import accepted_groups

DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
CHECKPOINT = ROOT / "checkpoints" / "reconstructor" / "best.pt"
GATE_PATH = ROOT / "checkpoints" / "gate.json"
BATCH = 4
ACCELERATION = 4
CENTER_FRACTION = 0.08
CASCADES = 5


def eval_pass(model, batches, device):
    """Single scored pass over the eval split: per-slice recon and baseline quality, residual, and contrast."""
    rows = {k: [] for k in ("recon_psnr", "recon_ssim", "recon_nmse",
                            "base_psnr", "base_ssim", "base_nmse", "residual")}
    contrasts = []
    with torch.inference_mode():
        for batch in batches:
            kspace = batch["masked_kspace"].to(device)
            mask = batch["mask"].to(device)
            scale = slice_scale(kspace)
            target = batch["target"].to(device) / scale
            image, residual = model.scored(kspace / scale.unsqueeze(1), mask)
            base = zero_filled(kspace) / scale
            rows["recon_psnr"].append(psnr(image, target).cpu())
            rows["recon_ssim"].append(ssim(image, target).cpu())
            rows["recon_nmse"].append(nmse(image, target).cpu())
            rows["base_psnr"].append(psnr(base, target).cpu())
            rows["base_ssim"].append(ssim(base, target).cpu())
            rows["base_nmse"].append(nmse(base, target).cpu())
            rows["residual"].append(residual.cpu())
            contrasts.extend(batch["contrast"])
    clear_cache(device)
    data = {k: torch.cat(v).numpy() for k, v in rows.items()}
    data["contrast"] = np.asarray(contrasts)
    return data


def representatives(quality, contrast):
    """Position of the slice nearest the median SSIM within each contrast."""
    picks = {}
    for name in sorted(set(contrast.tolist())):
        idx = np.flatnonzero(contrast == name)
        picks[name] = int(idx[np.argmin(np.abs(quality[idx] - np.median(quality[idx])))])
    return picks


def collate(dataset, indices):
    """Stack chosen dataset items into one batch of tensors."""
    items = [dataset[i] for i in indices]
    return {"masked_kspace": torch.stack([it["masked_kspace"] for it in items]),
            "mask": torch.stack([it["mask"] for it in items]),
            "target": torch.stack([it["target"] for it in items])}


def panel_rows(model, batch, names, certified, device):
    """Build (contrast, [zero-filled, recon, target], annotations) for each chosen slice."""
    with torch.inference_mode():
        kspace = batch["masked_kspace"].to(device)
        mask = batch["mask"].to(device)
        scale = slice_scale(kspace)
        target = batch["target"].to(device) / scale
        image, _ = model.scored(kspace / scale.unsqueeze(1), mask)
        base = zero_filled(kspace) / scale
        recon_ssim, base_ssim = ssim(image, target).cpu(), ssim(base, target).cpu()
        recon_psnr = psnr(image, target).cpu()
    rows = []
    for i, name in enumerate(names):
        verdict = "certified" if certified[i] else "flagged"
        tags = [f"SSIM {base_ssim[i]:.3f}",
                f"SSIM {recon_ssim[i]:.3f} | {recon_psnr[i]:.1f} dB | {verdict}", ""]
        imgs = [base[i].cpu().numpy(), image[i].cpu().numpy(), target[i].cpu().numpy()]
        rows.append((name, imgs, tags))
    clear_cache(device)
    return rows


def recon_panel(rows, path):
    """Render and save the per-contrast reconstruction panel."""
    import matplotlib.pyplot as plt
    titles = ("zero-filled", "reconstructed", "target")
    figure, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.7 * len(rows)), squeeze=False)
    for r, (name, imgs, tags) in enumerate(rows):
        vmax = float(imgs[2].max()) or 1.0
        for c, (title, image, tag) in enumerate(zip(titles, imgs, tags)):
            axis = axes[r][c]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if r == 0:
                axis.set_title(title, fontsize=11)
            if tag:
                axis.set_xlabel(tag, fontsize=9)
        axes[r][0].set_ylabel(name, fontsize=12)
    figure.suptitle("Reconstruction and gate certification (held-out eval)", y=1.0, fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    device = active_device()
    dataset = M4RawSlices(DATA_DIR, acceleration=ACCELERATION, center_fraction=CENTER_FRACTION)
    groups = scan_groups(dataset, SPLITS, SEED)
    workers = worker_count()
    batches = DataLoader(Subset(dataset, groups["eval"]), batch_size=BATCH, shuffle=False,
                         num_workers=workers, persistent_workers=workers > 0)

    model = Reconstructor(coils=4, cascades=CASCADES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["model"])
    model.eval()

    data = eval_pass(model, batches, device)
    print(f"zero-filled  : PSNR {data['base_psnr'].mean():6.2f} dB | SSIM {data['base_ssim'].mean():.3f} | NMSE {data['base_nmse'].mean():.4f}")
    print(f"reconstructed: PSNR {data['recon_psnr'].mean():6.2f} dB | SSIM {data['recon_ssim'].mean():.3f} | NMSE {data['recon_nmse'].mean():.4f}")

    residual, contrast = data["residual"], data["contrast"]
    error = np.clip(1.0 - data["recon_ssim"], 0.0, 1.0)
    keep = None
    if GATE_PATH.exists():
        gate = json.loads(GATE_PATH.read_text())
        epsilon = gate["epsilon"]
        keep = accepted_groups(residual, contrast, gate["thresholds"])
        aurocs = contrast_auroc(residual, error, contrast, epsilon)
        aurcs = contrast_aurc(residual, error, contrast)
        print(f"\ngate epsilon {epsilon} | certified-set risk holds when risk <= epsilon")
        for name in sorted(set(contrast.tolist())):
            pick = contrast == name
            taken = keep & pick
            certify = float(keep[pick].mean()) if pick.any() else 0.0
            risk = float(error[taken].mean()) if taken.any() else float("nan")
            print(f"{name:5s} certify {certify:5.1%} | risk {risk:5.3f} | AUROC {aurocs[name]:.2f} | AURC {aurcs[name]:.3f}")
        overall_risk = float(error[keep].mean()) if keep.any() else float("nan")
        print(f"overall certify {keep.mean():5.1%} | risk {overall_risk:5.3f}")
    else:
        print("\ngate.json not found - run calibrate.py for gate risk; panel verdicts show flagged")

    picks = representatives(data["recon_ssim"], contrast)
    names = list(picks)
    positions = [picks[name] for name in names]
    certified = keep[positions] if keep is not None else np.zeros(len(names), dtype=bool)
    batch = collate(dataset, [groups["eval"][p] for p in positions])
    rows = panel_rows(model, batch, names, certified, device)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    recon_panel(rows, ART_DIR / "recon_panel.png")
    print(f"wrote recon_panel.png to {ART_DIR}")