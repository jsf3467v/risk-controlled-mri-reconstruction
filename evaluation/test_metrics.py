"""Unit tests for reconstruction quality metrics."""

import torch

from src.metrics import psnr, ssim, nmse


def random_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 64, 64, generator=g)


def test_nmse_zero_when_identical():
    img = random_batch()
    assert nmse(img, img).abs().max() < 1e-6


def test_ssim_one_when_identical():
    img = random_batch()
    assert (ssim(img, img) - 1).abs().max() < 1e-4


def test_nmse_grows_with_error():
    img = random_batch()
    noisy = img + 0.1 * random_batch(seed=1)
    assert (nmse(img, noisy) > nmse(img, img)).all()


def test_psnr_drops_with_error():
    img = random_batch()
    noisy = img + 0.1 * random_batch(seed=2)
    assert (psnr(img, noisy) < psnr(img, img)).all()
