"""Shared scoring so training, evaluation, and calibration measure the same quantities."""
import numpy as np
import torch
import torch.nn.functional as F

from data_processing import slice_scale, clear_cache


def ssim(pred, target, window=7):
    """Per-sample structural similarity over a uniform window, scaled to each slice's dynamic range."""
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)
    pad = window // 2
    kernel = torch.ones(1, 1, window, window, device=pred.device) / (window ** 2)
    mu_p = F.conv2d(pred, kernel, padding=pad)
    mu_t = F.conv2d(target, kernel, padding=pad)
    var_p = (F.conv2d(pred * pred, kernel, padding=pad) - mu_p ** 2).clamp_min(0.0)
    var_t = (F.conv2d(target * target, kernel, padding=pad) - mu_t ** 2).clamp_min(0.0)
    cov = F.conv2d(pred * target, kernel, padding=pad) - mu_p * mu_t
    data_range = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
    denominator = (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


def psnr(pred, target):
    """Per-sample peak signal-to-noise ratio in decibels."""
    mse = ((pred - target) ** 2).mean(dim=(-2, -1))
    peak = target.amax(dim=(-2, -1)).clamp_min(1e-8)
    return 10 * torch.log10(peak ** 2 / (mse + 1e-12))


def nmse(pred, target):
    """Per-sample normalized mean squared error."""
    error = ((pred - target) ** 2).sum(dim=(-2, -1))
    energy = (target ** 2).sum(dim=(-2, -1)).clamp_min(1e-12)
    return error / energy


def slice_scores(model, loader, device):
    """Per-slice data-consistency residual, structural error, and contrast over a loader, in one forward pass each."""
    scores, errors, contrasts = [], [], []
    with torch.inference_mode():
        for batch in loader:
            kspace = batch["masked_kspace"].to(device)
            mask = batch["mask"].to(device)
            scale = slice_scale(kspace)
            pred, residual = model.scored(kspace / scale.unsqueeze(1), mask)
            target = batch["target"].to(device) / scale
            scores.append(residual.cpu())
            errors.append((1.0 - ssim(pred, target)).clamp(0.0, 1.0).cpu())
            contrasts.extend(batch["contrast"])
    clear_cache(device)
    return torch.cat(scores).numpy(), torch.cat(errors).numpy(), np.asarray(contrasts)


def auroc(scores, labels):
    """Rank-based area under the ROC curve for higher scores predicting the positive label."""
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(labels.size)
    ranks[order] = np.arange(1, labels.size + 1)
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def contrast_auroc(scores, errors, contrasts, epsilon):
    """Per-contrast AUROC of the residual against the error-over-epsilon label."""
    scores, errors, contrasts = np.asarray(scores), np.asarray(errors), np.asarray(contrasts)
    return {str(name): auroc(scores[contrasts == name], errors[contrasts == name] > epsilon)
            for name in np.unique(contrasts)}


def risk_coverage(scores, errors):
    """Selective risk at each coverage as the residual threshold loosens, scores ascending."""
    order = np.argsort(np.asarray(scores))
    ranked = np.clip(np.asarray(errors)[order], 0.0, 1.0)
    k = np.arange(1, len(ranked) + 1)
    return k / len(ranked), np.cumsum(ranked) / k


def aurc(scores, errors):
    """Area under the risk-coverage curve. Lower means the residual orders errors better."""
    if len(errors) == 0:
        return float("nan")
    return float(risk_coverage(scores, errors)[1].mean())


def contrast_aurc(scores, errors, contrasts):
    """Per-contrast area under the risk-coverage curve of the residual against structural error."""
    scores, errors, contrasts = np.asarray(scores), np.asarray(errors), np.asarray(contrasts)
    return {str(name): aurc(scores[contrasts == name], errors[contrasts == name])
            for name in np.unique(contrasts)}