"""Risk-vs-escalation frontier. Sweep the tolerance epsilon and, at each, recalibrate the
per-contrast gate on the calibrate split and measure realized accept rate and risk on eval. One
scored pass per split, then pure numpy over the same gate functions calibrate.py and eval.py use.
The README describes the frontier and how to read it.
"""
import sys
from pathlib import Path

import numpy as np
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
ART_DIR = ROOT / "artifacts"
CHECKPOINT = ROOT / "checkpoints" / "reconstructor" / "best.pt"
BATCH = 4
ACCELERATION = 4
CENTER_FRACTION = 0.08
CASCADES = 5
ALPHA = 0.1
MINIMUM = 50
CANDIDATES = 8
REFERENCE_EPSILON = 0.20
EPSILONS = np.round(np.linspace(0.05, 0.60, 23), 3)
TRIALS = 200
PALETTE = {"T1": "#0072B2", "T2": "#D55E00", "FLAIR": "#009E73"}


def split_scores(model, dataset, indices, device):
    """Per-slice residual score, structural error, and contrast for one split."""
    workers = worker_count()
    loader = DataLoader(Subset(dataset, indices), batch_size=BATCH, shuffle=False,
                        num_workers=workers, persistent_workers=workers > 0)
    return slice_scores(model, loader, device)


def study_labels(dataset, indices):
    """Study id for each scored slice, aligned to the split order, for subject-level resampling."""
    studies = dataset.scans.index.get_level_values("study").to_numpy()
    return np.array([studies[dataset.items[i][0]] for i in indices])


def resampled_risk(scores, errors, contrasts, studies, fraction, epsilon, alpha, minimum, candidates, trials, seed):
    """Realized certified-set risk over random subject-level calibrate/eval splits at one tolerance."""
    rng = np.random.default_rng(seed)
    names = np.unique(studies)
    risks = np.full(trials, np.nan)
    for t in range(trials):
        order = rng.permutation(len(names))
        held = set(names[order[:round(len(names) * fraction)]])
        cal = np.fromiter((s in held for s in studies), dtype=bool, count=len(studies))
        thresholds = mondrian_thresholds(scores[cal], errors[cal], contrasts[cal], epsilon, alpha, minimum, candidates)
        keep = accepted_groups(scores[~cal], contrasts[~cal], thresholds)
        if keep.any():
            risks[t] = errors[~cal][keep].mean()
    return risks


def resampling_figure(risks, epsilon, path):
    """Histogram of realized certified-set risk across resamples with the tolerance marked."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(risks, bins=24, color=PALETTE["T1"], alpha=0.85)
    ax.axvline(epsilon, color="crimson", linewidth=1.2, linestyle="--", label="requested tolerance")
    ax.set_xlabel("realized risk on held-out eval  (mean $1-$SSIM)")
    ax.set_ylabel("resamples")
    ax.set_title("Certified-set risk under resampling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def frontier(cal, ev, epsilons, alpha, minimum, candidates):
    """Recalibrate per contrast at each epsilon and record eval accept rate and realized risk.

    cal and ev are (scores, errors, contrasts) triples. Returns a structured array with one
    row per (epsilon, contrast) plus an 'overall' pseudo-contrast aggregating all slices.
    """
    cal_s, cal_e, cal_c = (np.asarray(x) for x in cal)
    ev_s, ev_e, ev_c = (np.asarray(x) for x in ev)
    names = sorted(set(ev_c.tolist()))
    rows = []
    for eps in epsilons:
        thresholds = mondrian_thresholds(cal_s, cal_e, cal_c, eps, alpha, minimum, candidates)
        keep = accepted_groups(ev_s, ev_c, thresholds)
        for name in names:
            pick = ev_c == name
            taken = keep & pick
            accept = float(keep[pick].mean()) if pick.any() else 0.0
            risk = float(ev_e[taken].mean()) if taken.any() else np.nan
            rows.append((float(eps), name, accept, risk))
        accept = float(keep.mean())
        risk = float(ev_e[keep].mean()) if keep.any() else np.nan
        rows.append((float(eps), "overall", accept, risk))
    return np.array(rows, dtype=[("epsilon", float), ("contrast", "U8"),
                                 ("accept", float), ("risk", float)])


def frontier_csv(table, path):
    """Tidy long-format dump of the swept frontier for the writeup."""
    lines = ["epsilon,contrast,accept,risk"]
    for r in table:
        risk = "" if np.isnan(r["risk"]) else f"{r['risk']:.6f}"
        lines.append(f"{r['epsilon']:.3f},{r['contrast']},{r['accept']:.6f},{risk}")
    path.write_text("\n".join(lines) + "\n")


def line_style(name):
    """Plot style for one contrast trace, emphasizing the overall aggregate."""
    overall = name == "overall"
    return dict(marker="o", markersize=3, linewidth=2.4 if overall else 1.6,
                color="black" if overall else PALETTE.get(name),
                linestyle="--" if overall else "-", label=name, zorder=3 if overall else 2)


def accept_panel(ax, table, order):
    """Acceptance-vs-tolerance panel: one trace per contrast plus the aggregate."""
    for name in order:
        rows = np.sort(table[table["contrast"] == name], order="epsilon")
        ax.plot(rows["epsilon"], rows["accept"], **line_style(name))
    ax.set_xlabel("requested tolerance  $\\epsilon$")
    ax.set_ylabel("accept rate  (1 = no escalation)")
    ax.set_title("Acceptance vs tolerance")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(title="contrast", fontsize=8, loc="upper left")


def risk_panel(ax, table, order, lim):
    """Realized-vs-requested-risk panel with the guarantee region shaded."""
    ax.plot(lim, lim, color="gray", linewidth=1, zorder=1)
    ax.fill_between(lim, lim, [lim[1]] * 2, color="green", alpha=0.06, zorder=0)
    ax.text(0.97 * lim[1], 0.5 * lim[1], "guarantee holds\n(realized $\\leq$ requested)",
            ha="right", va="center", fontsize=8, color="green")
    for name in order:
        rows = np.sort(table[table["contrast"] == name], order="epsilon")
        valid = ~np.isnan(rows["risk"])
        ax.plot(rows["epsilon"][valid], rows["risk"][valid], **line_style(name))
    ax.set_xlabel("requested tolerance  $\\epsilon$")
    ax.set_ylabel("realized risk on eval  (mean $1-$SSIM)")
    ax.set_title("Realized vs requested risk")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)


def frontier_figure(table, path, reference):
    """Two-panel frontier: acceptance and realized-vs-requested risk on held-out eval."""
    import matplotlib.pyplot as plt

    contrasts = [c for c in dict.fromkeys(table["contrast"].tolist()) if c != "overall"]
    order = contrasts + ["overall"]
    lim = [0.0, float(max(EPSILONS))]
    fig, (ax_acc, ax_risk) = plt.subplots(1, 2, figsize=(11, 4.4))
    accept_panel(ax_acc, table, order)
    risk_panel(ax_risk, table, order, lim)
    for ax in (ax_acc, ax_risk):
        ax.axvline(reference, color="crimson", linewidth=0.9, linestyle=":", zorder=1)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Cost-adaptive gate: risk-vs-escalation frontier (held-out eval)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    device = active_device()
    dataset = M4RawSlices(DATA_DIR, acceleration=ACCELERATION, center_fraction=CENTER_FRACTION)
    groups = scan_groups(dataset, SPLITS, SEED)

    model = Reconstructor(coils=4, cascades=CASCADES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["model"])
    model.eval()

    cal = split_scores(model, dataset, groups["calibrate"], device)
    ev = split_scores(model, dataset, groups["eval"], device)

    table = frontier(cal, ev, EPSILONS, ALPHA, MINIMUM, CANDIDATES)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ART_DIR / "risk_escalation_sweep.csv"
    png_path = ART_DIR / "risk_escalation_frontier.png"
    frontier_csv(table, csv_path)
    frontier_figure(table, png_path, REFERENCE_EPSILON)

    names = [c for c in dict.fromkeys(table["contrast"].tolist())]
    print(f"swept {len(EPSILONS)} tolerances in [{EPSILONS[0]}, {EPSILONS[-1]}] | alpha {ALPHA}")
    for name in names:
        rows = np.sort(table[table["contrast"] == name], order="epsilon")
        lifts = rows["epsilon"][rows["accept"] > 0.5]
        liftoff = f"accepts >50% from eps={lifts[0]:.3f}" if lifts.size else "never accepts >50%"
        print(f"  {name:8s} {liftoff}")
    print(f"wrote {csv_path.name} and {png_path.name} to {ART_DIR}")

    studies = np.concatenate([study_labels(dataset, groups["calibrate"]),
                              study_labels(dataset, groups["eval"])])
    pooled = (np.concatenate([cal[0], ev[0]]), np.concatenate([cal[1], ev[1]]),
              np.concatenate([cal[2], ev[2]]))
    fraction = SPLITS["calibrate"] / (SPLITS["calibrate"] + SPLITS["eval"])
    risks = resampled_risk(*pooled, studies, fraction, REFERENCE_EPSILON,
                           ALPHA, MINIMUM, CANDIDATES, TRIALS, SEED)
    valid = risks[~np.isnan(risks)]
    held = float((valid <= REFERENCE_EPSILON).mean()) if valid.size else float("nan")
    resample_path = ART_DIR / "risk_guarantee_resampling.png"
    resampling_figure(valid, REFERENCE_EPSILON, resample_path)
    print(f"resampled {TRIALS} splits at eps {REFERENCE_EPSILON} | accepted in {valid.size} | "
          f"risk <= eps in {held:5.1%} (target >= {1 - ALPHA:.0%}) | wrote {resample_path.name}")