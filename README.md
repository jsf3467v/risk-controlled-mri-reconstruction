[![CI](https://github.com/jsf3467v/risk-controlled-mri-reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/risk-controlled-mri-reconstruction/actions/workflows/ci.yml)

[![docker](https://github.com/jsf3467v/risk-controlled-mri-reconstruction/actions/workflows/docker-image.yml/badge.svg)](https://github.com/jsf3467v/risk-controlled-mri-reconstruction/actions/workflows/docker-image.yml)

# Risk-Controlled Certification for Accelerated Low-Field MRI Reconstruction

This project reconstructs accelerated low-field brain MRI scans and determines the reliability of each slice. A cost-effective reconstructor processes every slice. A risk-controlled gate then certifies the slices it deems reliable, providing a coverage guarantee, while flagging the others for review. The gate is the key innovation. The work focuses on the M4Raw low-field dataset, characterized by high noise levels where reliable quality control is essential.

## How the system works

Each slice traverses the system only once. A low-cost reconstructor processes the undersampled, single-repetition slice, producing a magnitude image in a single forward pass; here, 'low-cost' refers to computational expense, not image quality. The reconstructor is a CNN-based, end-to-end variational network that learns coil sensitivity maps and enforces sensitivity-weighted data consistency. The same pass also produces a data-consistency residual.

The gate interprets residual as an uncertainty score. It was calibrated using a separate set, where it adjusts one threshold per contrast to ensure the mean structural error of the certified set remains within a specified tolerance. This threshold is based on an empirical-Bernstein bound with a Bonferroni correction across a limited set of candidates, ensuring that the guarantee is probabilistic rather than heuristic.

From there, the decision is immediate. A slice whose residual falls below the contrast threshold is certified, and any slice above that threshold is flagged for review or further acquisition.

## The reconstructor

The reconstructor employs Sriram et al.'s end-to-end variational network, containing about 12.9 million parameters. Initially, a small network predicts coil sensitivity maps from the fully sampled central k-space, known as the autocalibration region, and then normalizes them so that the coil energy at each pixel sums to 1. This is followed by five cascades, each refining the k-space estimate through a learned data-consistency step tied to the measured lines and a CNN correction within the sensitivity domain. This process progressively aligns the estimate with both the measurements and the learned image prior. The final image is obtained by taking the root-sum-of-squares of the last cascade output, matching the magnitude domain used by the gate and metrics.

The uncertainty score represents the average absolute data-consistency gap across the measured lines. Calculating it requires no additional network or a second forward pass, ensuring that training, calibration, and evaluation all assess the same metric.

## The gate

For each contrast, the gate ranks calibration slices according to their scores and examines a small grid of acceptance fractions. At each point, it computes an empirical-Bernstein upper bound on the average structural error of the accepted slices. This bound depends on the errors' sample mean and variance, becoming tighter when errors are small and consistent. A Bonferroni correction across the grid ensures the overall guarantee stays valid, even with data-driven threshold selection. The gate then picks the largest threshold whose bound stays below the set tolerance, accepting as many slices as the guarantee allows.

The primary factor is the grid size. Using a coarser grid lowers the multiplicity penalty, tightens the bound, and boosts acceptance, but decreases threshold resolution. This run uses eight grid points. If calibration slices are too few, the gate defaults to full escalation for maximum safety.

Thresholds are set individually for each contrast rather than applied across all slices at once. This approach, inspired by Mondrian's concept, reveals the true behavior for each contrast instead of allowing an easier contrast to mask a more difficult one.

## Data

M4Raw is a low-field brain MRI collection acquired at 0.3 tesla with four receive coils and three contrasts, T1, T2, and FLAIR. Each scan has several repetitions, and the clean target for a slice is the average across them. Input slices are undersampled at four-fold acceleration ($4\times$), except for a fully sampled center. The data is split at the subject level into train, monitor, calibrate, and eval, so no subject appears in more than one group.

## Results

On the held-out eval split, the reconstructor significantly outperforms the naive baseline. The zero-filled baseline achieves 24.66 dB PSNR, 0.639 SSIM, and 0.0713 NMSE, while the reconstructor reaches 31.65 dB, 0.843 SSIM, and 0.0149 NMSE.

The panel below shows the median slice for each contrast. Each row displays the zero-filled input, the reconstruction, and the target, with the gate verdict for the reconstruction.

[![Per-contrast reconstruction and gate certification on the held-out eval split](artifacts/recon_panel.png)](artifacts/recon_panel.png)

The gate ran at a tolerance of 0.20 on mean $1 - \mathrm{SSIM}$. The table below reports the certified fraction and the realized risk per contrast on held-out data.

| contrast | certified | realized risk  | AUROC     | AURC  |
| -------- | --------- | -------------- | --------- | ----- |
| T1       | 100%      | 0.119          | undefined | 0.105 |
| T2       | 100%      | 0.160          | 0.69      | 0.153 |
| FLAIR    | 0%        | not applicable | 0.78      | 0.171 |
| overall  | 66.7%     | 0.139          |           |       |

Every certified set maintains its risk below the 0.20 tolerance level. T1 is at 0.119, T2 at 0.160, and the overall certified set is at 0.139. This summarizes the main outcome. The coverage guarantee extends from calibration through to held-out data.

AUROC assesses how effectively the residual score distinguishes slices that exceed the tolerance from those that do not. An AUROC of 0.5 indicates performance no better than random chance, while 1.0 signifies perfect separation. Here, T2 achieves 0.69 and FLAIR 0.78, showing that the residual ranking generally places failing slices above passing ones; this is especially noticeable in FLAIR. The T1 AUROC is undefined because none of the T1 slices surpass the tolerance threshold, meaning there are no failures to rank and indicating stable T1 reconstruction rather than a problem. FLAIR risk is not shown because no FLAIR slices are certified, so there is no certified set available for evaluation.

AURC complements AUROC by measuring the area under the risk-coverage curve, indicating how effectively residual orderings separate errors at each acceptance level, with lower values being better. AURC remains defined for T1 at 0.105 even when AUROC is not, and it orders the contrasts by difficulty as T1, then T2, and finally FLAIR, aligning with the certification sequence.

Sweeping the tolerance from 0.05 to 0.60 shows the full tradeoff. The left panel shows acceptance rising as the cutoff loosens. The right panel shows realized risk remaining within the guarantee region across the entire range.

[![Risk versus escalation frontier on the held-out eval split](artifacts/risk_escalation_frontier.png)](artifacts/risk_escalation_frontier.png)

Each contrast reaches the halfway point in acceptance once its specific tolerance threshold is exceeded. T1 reaches it at 0.150, T2 at 0.200, and FLAIR at 0.225. This sequence mirrors the increasing physical difficulty of the contrasts at low field, with T1 being the easiest and FLAIR the hardest. Throughout all tolerance levels, the actual risk remains below the target, ensuring the guarantee is valid across the entire operating range, not just at a single point.

## Robustness

The guarantee should remain valid across multiple resampling processes, not just after a single calibration and evaluation split. To verify this, scores are initially calculated once. Then, the calibration and evaluation datasets are resampled 200 times at the subject level. For each resample, the gate is refitted, and the actual risk of the certified set is recorded on the held-out side. Since each split maintains entire subjects together, no patient crosses between calibration and evaluation sets.

[![Certified-set risk across 200 subject-level resamples, with the tolerance marked](artifacts/risk_guarantee_resampling.png)](artifacts/risk_guarantee_resampling.png)

Every draw accepted slices, and the realized risk of the certified set remained at or below the tolerance each time. The risk clusters near 0.145 and never approaches the 0.20 line, so the guarantee held in every resample against a 90 percent target. The single-split result is not a lucky draw.

## Limitations and future work

The shipped system serves as the reconstructor and the gate. The escalation stage that would repair flagged slices is not included, which is a measured outcome rather than a gap. Two escalation models were tested. One was a self-supervised denoiser trained on independent repetition pairs. The other was a supervised refiner trained against the clean multi-repetition target. Neither improved the flagged FLAIR slices. The low-cost reconstructor already removes most recoverable noise from a single low-field scan and is, in fact, cleaner than the multi-repetition reference it is scored against, so a single-contrast model adds nothing. The flagged contrast sits just above the operating threshold, near 0.225, leaving little headroom. Noisier acquisitions are a different story. The system flags those slices correctly, which is the intended behavior.


A few approaches could still improve the flagged contrast. The most straightforward is multi-contrast reconstruction. The T1 and T2 images of the same subject reconstruct well and share anatomy with FLAIR. Conditioning FLAIR on them could provide information that a single contrast cannot. This would first require aligning the contrasts to the same slice, which was beyond the scope here. A more complex option is a generative prior that combines a diffusion model with data consistency and a null-space hallucination metric that measures how faithfully the generated structures match the true anatomy rather than how well they are hidden. Training such a model would require physics-aware data augmentation, since the dataset is small. A simpler alternative is to improve acquisition practices by repeating and averaging flagged slices to increase signal without relying on a model.

## Running the model

The scripts run in order, and each step writes what the next one needs.

```
python data/data_processing.py   # device preflight, run once before a long job
python src/train.py              # train the low-cost reconstructor
python evaluation/calibrate.py   # fit the per-contrast gate
python evaluation/eval.py        # certification report and reconstruction panel
python evaluation/sweep.py       # risk and certification frontier
```

The repository includes the data pipeline but not the M4Raw scans or the trained reconstructor weights. Download M4Raw, place its `.h5` files in `data/`, and run training first so that `checkpoints/reconstructor/best.pt` exists before calibration and evaluation. The gate file at `checkpoints/gate.json` is produced during calibration.

Run the preflight first. It exercises the complex k-space paths on the active device and reports pass or fail per operation. This matters most on Apple silicon, where support for complex FFT has varied across PyTorch versions. On CUDA or CPU it is a quick confirmation. Training is resumable, so if it stops you can rerun it and it continues from the latest checkpoint.

A containerized run that covers calibration, evaluation, the sweep, and the test suite is described in `DOCKER.md`.

## Repository layout

```
data/
  data_processing.py             k-space transforms, undersampling, clean targets, device preflight
src/
  reconstructor.py               the low-cost end-to-end variational network
  gate.py                        empirical-Bernstein risk control and the certify decision
  metrics.py                     shared SSIM, PSNR, NMSE, AUROC, and AURC
  train.py                       training for the reconstructor
evaluation/
  calibrate.py                   fit the per-contrast gate on the calibration split
  eval.py                        certification report and reconstruction panel
  sweep.py                       risk frontier and the resampling check
tests/
  test_gate.py                   risk-control gate behavior
  test_metrics.py                scoring function behavior
artifacts/
  recon_panel.png                per-contrast reconstruction and gate panel
  risk_escalation_frontier.png   acceptance and realized risk across tolerances
  risk_guarantee_resampling.png  certified-set risk across resampled splits
  risk_escalation_sweep.csv      swept tolerances in long format
  repetition_snr.png             repetition signal-to-noise from the exploratory analysis
  undersampling_preview.png      undersampling pattern preview
checkpoints/
  gate.json                      fitted per-contrast thresholds
EDA/
  EDA.ipynb                      exploratory analysis of the dataset
Dockerfile, docker-compose.yml, DOCKER.md   containerized evaluation
requirements.txt, requirements-dev.txt       runtime and test dependencies
pytest.ini                                    test import paths
.github/workflows/                            native and container continuous integration
```

## Reproducibility

Randomness is seeded where it is generated. All paths are relative, and the code runs unchanged on Apple silicon, CUDA, or CPU, so a run on one machine reproduces another.

## References

The shipped design draws on the following work.

1. M4Raw is a multi-contrast, multi-repetition, multi-channel low-field brain k-space dataset. Lyu et al., Scientific Data, 2023.
2. End-to-end variational networks are the reconstruction backbone used here. Sriram et al., MICCAI, 2020.
3. The fastMRI dataset and benchmarks established the accelerated MRI reconstruction setup. Knoll et al., Radiology Artificial Intelligence, 2020.
4. SSIM measures structural similarity and defines the error used here. Wang et al., IEEE Transactions on Image Processing, 2004.
5. Empirical Bernstein bounds with sample-variance penalization give the gate its tail bound. Maurer and Pontil, COLT, 2009.
6. Distribution-free risk-controlling prediction sets frame the certify decision. Bates et al., Journal of the ACM, 2021.
7. Learn then Test calibrates predictive algorithms to achieve risk control. Angelopoulos et al., 2021.
8. Mondrian confidence machines are the per-group idea behind the per-contrast thresholds. Vovk et al., 2003.

The escalation study and the future-work directions draw on the following.

9. Noise2Noise learns image restoration without clean data. Lehtinen et al., ICML, 2018.
10. Rep2Rep extends Noise2Noise to repeated low-field MRI acquisitions. Janjušević et al., Magnetic Resonance in Medicine, 2026.
11. Physics-aware data augmentation helps deep accelerated MRI with limited data. Fabian et al., ICML, 2021.
12. The null-space hallucination map measures invented structure in tomographic reconstruction. Bhadra et al., IEEE Transactions on Medical Imaging, 2021.

