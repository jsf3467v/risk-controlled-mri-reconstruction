"""M4Raw reconstruction data: k-space transforms, undersampling, and noisy/clean slice pairs."""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEED = 0
CONTRASTS = ("T1", "T2", "FLAIR")
SPLITS = {"train": 0.5, "monitor": 0.1, "calibrate": 0.3, "eval": 0.1}


def active_device():
    """Fastest available torch device, portable across machines."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_cache(device):
    """Release cached allocator memory on the active accelerator."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def worker_count():
    """Portable DataLoader worker count, capped to spare cores and bound prefetch memory."""
    return min(8, max((os.cpu_count() or 1) - 1, 0))


def fft2c(image):
    """Centered orthonormal 2D FFT over the last two axes."""
    axes = (-2, -1)
    shifted = torch.fft.ifftshift(image, dim=axes)
    spectrum = torch.fft.fft2(shifted, dim=axes, norm="ortho")
    return torch.fft.fftshift(spectrum, dim=axes)


def ifft2c(spectrum):
    """Centered orthonormal 2D inverse FFT over the last two axes."""
    axes = (-2, -1)
    shifted = torch.fft.ifftshift(spectrum, dim=axes)
    image = torch.fft.ifft2(shifted, dim=axes, norm="ortho")
    return torch.fft.fftshift(image, dim=axes)


def rss(coil_images):
    """Root-sum-of-squares coil combination over the coil axis."""
    return torch.sqrt((coil_images.abs() ** 2).sum(dim=-3).clamp_min(1e-12))


def zero_filled(masked_kspace):
    """Zero-filled magnitude image: inverse transform then coil combine."""
    return rss(ifft2c(masked_kspace))


def slice_scale(masked_kspace):
    """Per-sample magnitude scale from the zero-filled image, guarded against zeros."""
    peak = zero_filled(masked_kspace).amax(dim=(-2, -1)).clamp_min(1e-8)
    return peak.view(-1, 1, 1)


def preflight():
    """Run the complex k-space paths on the active device and report pass or fail per operation."""
    device = active_device()
    image = torch.randn(1, 4, 64, 64, dtype=torch.complex64, device=device)
    fft_ok = torch.allclose(ifft2c(fft2c(image)), image, atol=1e-4)
    view_ok = torch.equal(torch.view_as_complex(torch.view_as_real(image).contiguous()), image)
    combine_ok = bool(torch.isfinite(rss(image)).all())
    clear_cache(device)
    return {"device": device.type, "fft_roundtrip": fft_ok,
            "view_complex": view_ok, "coil_combine": combine_ok}


def cartesian_mask(width, acceleration, center_fraction, rng):
    """fastMRI-style 1D phase-encode mask with a fully sampled center, returned as floats."""
    center = round(width * center_fraction)
    span = max(width - center, 1)
    prob = float(np.clip((width / acceleration - center) / span, 0.0, 1.0))
    mask = rng.random(width) < prob
    start = (width - center) // 2
    mask[start:start + center] = True
    return torch.from_numpy(mask).to(torch.float32)


def scan_index(folder):
    """Table of M4Raw files parsed into study, contrast, and repetition."""
    rows = []
    for path in sorted(Path(folder).rglob("*.h5")):
        study, _, token = path.stem.partition("_")
        contrast = next((c for c in CONTRASTS if token.upper().startswith(c)), None)
        if contrast is None:
            continue
        rows.append((str(path), study, contrast, int(token[len(contrast):])))
    return pd.DataFrame(rows, columns=["path", "study", "contrast", "rep"])


def kspace_slice(path, z):
    """Complex multi-coil k-space for one slice of one repetition file."""
    with h5py.File(path, "r") as handle:
        return torch.from_numpy(handle["kspace"][z]).to(torch.complex64)


def rss_slices(paths, z):
    """Per-repetition root-sum-of-squares slices stacked along a new axis."""
    images = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            images.append(torch.from_numpy(handle["reconstruction_rss"][z]).float())
    return torch.stack(images)


def volume_depth(path):
    """Slice count shared by k-space and reconstruction in one file."""
    with h5py.File(path, "r") as handle:
        return min(handle["kspace"].shape[0], handle["reconstruction_rss"].shape[0])


def scan_groups(dataset, fractions, seed):
    """Disjoint subject-level item indices per named fraction, keeping every study wholly in one group."""
    study = dataset.scans.index.get_level_values("study")
    studies = study.unique().to_numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(studies))
    edges = np.cumsum([round(len(order) * f) for f in fractions.values()])[:-1]
    owner = {studies[s]: name for name, chunk in zip(fractions, np.split(order, edges)) for s in chunk}
    groups = {name: [] for name in fractions}
    for i, (scan, _) in enumerate(dataset.items):
        groups[owner[study[scan]]].append(i)
    return groups


class M4RawSlices(Dataset):
    """Slices paired as undersampled single-repetition input and multi-repetition clean target."""

    def __init__(self, folder, acceleration=4, center_fraction=0.08, seed=SEED):
        index = scan_index(folder)
        self.scans = index.sort_values("rep").groupby(["study", "contrast"])["path"].apply(list)
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.seed = seed
        self.items = [(s, z) for s, paths in enumerate(self.scans)
                      for z in range(volume_depth(paths[0]))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        scan, z = self.items[idx]
        paths = self.scans.iloc[scan]
        kspace = kspace_slice(paths[0], z)
        images = rss_slices(paths, z)
        rng = np.random.default_rng((self.seed, idx))
        mask = cartesian_mask(kspace.shape[-1], self.acceleration, self.center_fraction, rng)
        study, contrast = self.scans.index[scan]
        return {
            "masked_kspace": kspace * mask,
            "mask": mask,
            "target": images.mean(dim=0),
            "study": study,
            "contrast": contrast,
        }

if __name__ == "__main__":
    report = preflight()
    print(f"device {report['device']}")
    print(f"fft round-trip   {'ok' if report['fft_roundtrip'] else 'FAIL'}")
    print(f"real-complex view {'ok' if report['view_complex'] else 'FAIL'}")
    print(f"coil combine     {'ok' if report['coil_combine'] else 'FAIL'}")