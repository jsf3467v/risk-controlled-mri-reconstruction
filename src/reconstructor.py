"""Cheap reconstruction stage. An end-to-end variational network (E2E-VarNet, Sriram et al. 2020)
incorporates learned sensitivity-map estimation and sensitivity-weighted (SENSE) soft data consistency.
The README explains the cascade update, sensitivity maps, and the data-consistency residual that
the gate calibrates on.

"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT))

from data_processing import fft2c, ifft2c, rss


def as_channels(x):
    """Complex image stack (b, c, h, w) as interleaved real channels (b, 2c, h, w)."""
    real = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
    return real.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])


def as_complex(x):
    """Interleaved real channels (b, 2c, h, w) back to a complex image stack (b, c, h, w)."""
    b, ch, h, w = x.shape
    folded = x.reshape(b, ch // 2, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
    return torch.view_as_complex(folded)


def center_mask(kspace, fraction):
    """k-space with all but the fully sampled central (ACS) phase-encode lines zeroed."""
    width = kspace.shape[-1]
    center = round(width * fraction)
    start = (width - center) // 2
    keep = torch.zeros(width, device=kspace.device)
    keep[start:start + center] = 1.0
    return kspace * keep


class ConvBlock(nn.Module):
    """Two instance-normalized convolutions with leaky activations."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.body(x)


class Unet(nn.Module):
    """Compact U-Net with average-pool downsampling and nearest-neighbor upsampling."""

    def __init__(self, in_ch, out_ch, base, depth):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        chans = [base * 2 ** i for i in range(depth)]
        prev = in_ch
        self.downs = nn.ModuleList()
        for ch in chans:
            self.downs.append(ConvBlock(prev, ch))
            prev = ch
        self.mid = ConvBlock(prev, prev * 2)
        prev = prev * 2
        self.ups = nn.ModuleList()
        for ch in reversed(chans):
            self.ups.append(ConvBlock(prev + ch, ch))
            prev = ch
        self.head = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for up in self.ups:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = up(torch.cat([x, skip], dim=1))
        return self.head(x)


class NormUnet(nn.Module):
    """U-Net wrapped in per-sample standardization for stable training across slice scales."""

    def __init__(self, in_ch, out_ch, base, depth):
        super().__init__()
        self.unet = Unet(in_ch, out_ch, base, depth)

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return self.unet((x - mean) / std) * std + mean


class SensitivityNet(nn.Module):
    """Estimate normalized coil sensitivity maps from the ACS region, per coil, end-to-end."""

    def __init__(self, base, depth):
        super().__init__()
        self.unet = NormUnet(2, 2, base, depth)

    def forward(self, masked_kspace, fraction):
        coils = ifft2c(center_mask(masked_kspace, fraction))
        b, c, h, w = coils.shape
        refined = as_complex(self.unet(as_channels(coils.reshape(b * c, 1, h, w))))
        maps = refined.reshape(b, c, h, w)
        return maps / rss(maps).unsqueeze(1).clamp_min(1e-12)


class VarNetBlock(nn.Module):
    """One cascade. SENSE-domain CNN refinement and a learned soft data-consistency step."""

    def __init__(self, base, depth):
        super().__init__()
        self.refine = NormUnet(2, 2, base, depth)
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_reduce(self, kspace, maps):
        """Sensitivity-weighted coil combination to a single complex image."""
        return (ifft2c(kspace) * maps.conj()).sum(dim=-3, keepdim=True)

    def sens_expand(self, image, maps):
        """Single complex image back to multi-coil k-space through the sensitivity maps."""
        return fft2c(image * maps)

    def forward(self, kspace, measured, mask, maps):
        refined = self.sens_expand(as_complex(self.refine(as_channels(self.sens_reduce(kspace, maps)))), maps)
        soft_dc = self.dc_weight * mask * (kspace - measured)
        return kspace - soft_dc - refined


class Reconstructor(nn.Module):
    """Cheap stage: E2E-VarNet producing a magnitude image and its data-consistency residual."""

    def __init__(self, coils, cascades=5, chans=18, sens_chans=8, depth=4, center_fraction=0.08):
        super().__init__()
        self.center_fraction = center_fraction
        self.sense = SensitivityNet(sens_chans, depth)
        self.blocks = nn.ModuleList(VarNetBlock(chans, depth) for _ in range(cascades))

    def cascade(self, masked_kspace, mask):
        """Run every cascade and return the refined multi-coil k-space."""
        maps = self.sense(masked_kspace, self.center_fraction)
        view = mask.view(mask.shape[0], 1, 1, -1).to(masked_kspace.real.dtype)
        kspace = masked_kspace
        for block in self.blocks:
            kspace = block(kspace, masked_kspace, view, maps)
        return kspace

    def forward(self, masked_kspace, mask):
        return rss(ifft2c(self.cascade(masked_kspace, mask)))

    def scored(self, masked_kspace, mask):
        """Reconstructed magnitude image and its per-sample data-consistency residual."""
        kspace = self.cascade(masked_kspace, mask)
        image = rss(ifft2c(kspace))
        view = mask.view(mask.shape[0], 1, 1, -1).to(masked_kspace.real.dtype)
        gap = (kspace - masked_kspace).abs() * view
        residual = gap.sum(dim=(-3, -2, -1)) / view.expand_as(gap).sum(dim=(-3, -2, -1)).clamp_min(1.0)
        return image, residual


if __name__ == "__main__":
    coils = 4
    model = Reconstructor(coils, cascades=5)
    sample = torch.randn(2, coils, 256, 256, dtype=torch.complex64)
    sample_mask = (torch.rand(2, 256) > 0.5).float()
    output = model(sample, sample_mask)
    scored_image, score = model.scored(sample, sample_mask)
    match = torch.allclose(output, scored_image, atol=1e-5)
    print("output:", tuple(output.shape), "| residual:", tuple(score.shape),
          "| scored matches forward:", match,
          "| params:", sum(p.numel() for p in model.parameters()))