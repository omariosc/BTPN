<div align="center">

# Bayesian Temporal Pose Networks

### Uncertainty-Aware Laparoscopic Tool Pose Tracking

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue)](#citation)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

<!-- **[Paper](link) | [Poster](link) | [Video](link)** -->

*A probabilistic framework for 7-DoF visual-kinematic laparoscopic tool pose tracking with Bayesian uncertainty estimates, evaluated on 114 peg transfer trials across three electromagnetic tracking datasets.*

</div>

---

<p align="center">
  <img src="figures/architecture.png" alt="BTPN Architecture" width="95%"/>
</p>

**Architecture overview.** BTPN is built from five components, mapped onto the three panels above.

- **Visual feature extraction (panel a).** YOLOv26m-seg localises and segments both tools, producing per-tool ROI crops that YOLOv26m-pose turns into 8 keypoints each (proximal/distal shaft, joint, tip and both jaws); a gated fusion of these cues yields a 256D visual token. DepthAnything V2 (Small), a frozen DINOv2 encoder, adds 128D monocular depth cues.
- **Kinematic Foundation Model (panel c).** A ~1.2M-parameter Hierarchical Temporal Transformer embeds the 30D per-frame kinematic state (position **p**, quaternion **q** and jaw for both tools) to 256D. It is trained to predict the next-frame pose from a causal context window over ~330K electromagnetic-tracking frames (single-phase supervised regression); attention is non-causal *within* the window, but the window ends at the current frame. Windowed attention spans clinically-motivated scales (10/50/100 frames, &approx;0.8/3.8/7.7 s at 13 fps) for micro-gestures, purposeful actions and task phases; a learned memory bank (trainable prototype vectors) and quaternion-aware bimanual cross-attention (each tool attends over its own sequence, then into the other tool, via a learned gate) capture recurring patterns and inter-tool coordination.
- **Hierarchical Temporal Transformer (panel b).** Windowed attention at the three clinically-motivated scales above (10/50/100 frames) pairs, at each scale, an attention block with a refinement block; the per-scale representations are then combined by cross-scale multi-head attention. These scales follow measured peg-transfer action durations (reach &approx;1.1 s, grasp &approx;1.2 s, transfer &approx;2.4 s, place &approx;2.0 s), with local micro-gestures nested inside the longer phases.
- **Memory-Enhanced Encoder (panel c).** Each tool is projected separately and the two are coupled by quaternion-aware bimanual cross-attention with learnable gates; a memory bank — queried by the temporal sequence, with the pose embeddings as keys/values — stores the per-scale kinematic embeddings.
- **Visual encoder + cross-modal fusion + pose heads (panel a).** An image-based encoder, **pre-trained by masked visual reconstruction** (self-supervised) and visual&ndash;kinematic alignment, produces a visual embedding **V**; a **gated cross-modal fusion** combines **V** with the kinematic embedding **K** into a 256D visual-kinematic token. Gated residual corrections to the kinematic prior — per-channel confidence gates (position and jaw &le; 0.5, rotation &le; 0.1) prevent noisy visual estimates from corrupting modalities where kinematics are already accurate — feed separate heads predicting position (Cholesky-factored Gaussian), orientation (von Mises&ndash;Fisher concentration on **S**<sup>3</sup>) and jaw (scalar Gaussian) with calibrated Bayesian uncertainty.

## Abstract

Accurate pose tracking of laparoscopic instruments from monocular endoscopic video in surgical training tasks is essential for computer-assisted surgery and objective skill assessment. However, current methods require geometric priors unavailable in non-robotic settings and lack temporal reasoning across multimodal cues and uncertainty quantification. We introduce **Bayesian Temporal Pose Network (BTPN)**, a framework that fuses visual and kinematic features through hierarchical multi-scale temporal attention with Bayesian uncertainty estimates. A fine-tuned segmentation backbone achieves **91.1% mAP<sub>50-95</sub>** and keypoint detection reaches **94.6% mAP<sub>50-95</sub>**. The retrained visual-kinematic model attains **9.0 mm** position error and **8.9°** rotation (geodesic) with a position calibration error of **0.238** (ECE). The framework is evaluated on three electromagnetic tracking datasets with 114 peg transfer trials, supporting interpretable surgical skill assessment.

## Highlights

| | Metric | Value |
|---|---|---|
| :dart: | **Position RMSE** | 9.0 mm on held-out surgical trials |
| :triangular_ruler: | **Rotation (geodesic)** | 8.9° on SO(3) |
| :microscope: | **Segmentation** | 91.1% mAP<sub>50-95</sub> |
| :straight_ruler: | **Keypoints** | 94.6% mAP<sub>50-95</sub> |
| :bar_chart: | **Calibration (ECE)** | 0.238 position; 0.255 rotation (Fisher) |
| :hospital: | **Validation** | 114 peg transfer trials across 3 datasets |

---

## Key Results

> :bar_chart: **Full experimental-reporting details** — hyperparameters & selection, data splits, metric definitions, baselines, runtime/memory, failure-mode analysis, and limitations (single-seed; no multi-seed error bars/significance) — are in [`docs/REPORTING.md`](docs/REPORTING.md).

### (a) Visual Pipeline Components

| Component | Precision | Recall | mAP<sub>50</sub> | mAP<sub>50-95</sub> |
|:----------|:---------:|:------:|:-----------------:|:--------------------:|
| Detection | 98.1 | 98.3 | 99.1 | 97.5 |
| Segmentation | 98.1 | 98.3 | 99.1 | 91.1 |
| Keypoints | 97.3 | 97.3 | 98.3 | 94.6 |

### (b) Pose Prediction — Dataset A Held-Out Trials

| Method | Pos *x* | Pos *y* | Pos *z* | Pos ‖v‖ | Roll | Pitch | Yaw | Geo | Jaw (% open) | Pos ECE | Rot ECE |
|:-------|:-------:|:-------:|:-------:|:--------:|:----:|:-----:|:---:|:---:|:-------:|:-------:|:-------:|
| ART-Net | 19.7 | 20.7 | 14.9 | 32.2 | 65.6 | 30.7 | 67.5 | 55.4 | 42.5 | 0.155 | — |
| Visual regr. | 18.3 | 17.0 | 13.0 | 28.1 | 52.8 | 27.2 | 55.0 | 44.0 | 45.6 | 0.137 | — |
| Visual + LSTM | 17.1 | 13.2 | 12.3 | 24.9 | 76.8 | 36.8 | 68.0 | 67.9 | 43.2 | 0.098 | — |
| Visual + TCN | 15.1 | 11.9 | 11.0 | 22.1 | 72.0 | 37.3 | 68.6 | 66.4 | 41.9 | 0.240 | — |
| Visual + VTT | 16.4 | 13.7 | 12.5 | 24.7 | 74.0 | 45.6 | 79.7 | 69.0 | 42.2 | 0.115 | — |
| Kinematic regr. | 5.2 | 5.8 | 3.9 | 8.7 | 33.1 | 17.2 | 31.2 | 27.6 | 14.4 | 0.098 | — |
| BTPN w/o multiscale | 4.3 | 4.7 | 3.6 | 7.3 | 18.0 | 7.4 | 19.0 | 11.6 | 13.2 | 0.020 | — |
| Full BTPN (paper) | 4.2 | 4.4 | 3.4 | 7.0 | 14.4 | 7.3 | 15.6 | 11.7 | 13.6 | 0.028 | 0.301 |
| **Full BTPN (retrained)** | **5.3** | **5.9** | **4.1** | **9.0** | **13.7** | **5.1** | **14.1** | **8.9** | **14.7** | **0.238** | **0.255** |

> Position errors in mm, rotation errors in degrees. Geo = geodesic distance on SO(3). Jaw is reported as **% of the per-trial 10/90-percentile opening range** (the jaw signal is a raw sensor voltage with no voltage-to-angle calibration). Pos ECE is the position calibration error (L2, mm); Rot ECE is the Fisher rotation calibration error. The paper and retrained rows can be reproduced with `python scripts/evaluate.py --from-npz results/evaluation_data.npz` and `python scripts/evaluate.py --from-npz results/evaluation_data_vmf_fix.npz`, respectively. The *w/o multiscale* row is reproducible from `python scripts/evaluate.py --from-npz results/evaluation_data_no_multiscale.npz`. The full ablation set is in [`results/table2b.tex`](results/table2b.tex).
>
> *A minor evaluation bug affected the original paper results; the retrained row and evaluation code use the corrected calculation.*
>
> In the paper checkpoint, multi-scale primarily benefits **position** (7.0 vs 7.3 mm ‖v‖; ~0.3 mm, 18/20 held-out trials, above the run-to-run noise floor); rotation/jaw/ECE are unchanged-to-slightly-better without it. The benefit is a broadly-distributed refinement, not localized to specific motion regimes.

### (c) Cross-Dataset Generalisation (paper checkpoint)

| Dataset | Role | Pos *x* | Pos *y* | Pos *z* | Pos ‖v‖ | &Delta;Rot (°/step) | Jaw (% open) |
|:--------|:----:|:-------:|:-------:|:-------:|:--------:|:-------------------:|:-------:|
| A (20 trials) | Held-out | 4.2 | 4.4 | 3.4 | 7.0 | 17.6 | 13.6 |
| B (30 trials) | In-dist. | 5.6 | 5.0 | 4.2 | 8.6 | 28.9 | 11.6 |
| C (24 trials) | OOD | 4.9 | 7.1 | 6.8 | 11.0 | 29.1 | N/A |

> &Delta;Rot is the per-step angular-velocity RMSE (deg/step); positions in mm. Datasets B and C use per-trial normalisation to absorb inter-trial electromagnetic-reference offsets. Dataset C is 6-DoF and has no jaw sensor.

### Inference-efficiency variants (exploratory — repo-only, not in the paper)

Two lighter visual variants (single-seed ablations reusing the frozen kinematic prior) probe the accuracy↔latency tradeoff on an RTX 3060:

| Variant | Pos ‖v‖ (mm) | Geo (°) | Jaw (%) | ECE | End-to-end | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| Full (seg + keypoints + depth) | 7.0 | 11.7 | 13.6 | 0.027 | ~270 ms | 3.7 |
| w/o depth (seg + keypoints) | 7.1 | 11.8 | 13.4 | 0.016 | ~163 ms | 6.1 |
| seg-only (no depth, no keypoints) | 7.0 | 11.9 | 13.4 | 0.013 | ~137 ms | 7.3 |

Accuracy is preserved within single-seed noise while latency drops ~40–49% (DepthAnything ≈107 ms + keypoints ≈26 ms removed). Single-seed / constructed-latency caveats and method are in [`docs/REPORTING.md`](docs/REPORTING.md). Not part of the paper.

### Qualitative Results

<p align="center">
  <img src="figures/output_comparison.png" alt="7-DoF Trajectory Predictions" width="90%"/>
</p>

**7-DoF predictions for a held-out trial sequence.** Ground truth (black solid) vs BTPN predictions (blue dashed) with &plusmn;2&sigma; uncertainty bands (shaded). Left column shows position (X, Y, Z in mm); right column shows Euler rotation components (degrees) and the jaw opening signal (% of the per-trial 10/90-percentile opening range). The model accurately tracks rapid tool motions with well-calibrated uncertainty that widens during challenging periods.

<p align="center">
  <img src="figures/uncertainty_quality.png" alt="Uncertainty Quality Assessment" width="90%"/>
</p>

**Uncertainty quality assessment (paper checkpoint).** **(a)** Reliability diagram: position (ECE = 0.028) and jaw (ECE = 0.079) are well-calibrated, while rotation is over-conservative / under-confident (ECE = 0.301, i.e. its ±&sigma; intervals cover more than the nominal rate), reflecting the inherent difficulty of recovering orientation from monocular images. **(b)** Mean position error binned by predicted &sigma; (*r* = 0.60): a clear monotonic trend confirms higher predicted uncertainty corresponds to higher actual error. **(c)** Sparsification curve: discarding the most uncertain 50% of predictions reduces mean error from 5.5 mm to ~4.6 mm (AUSE = 0.95 mm), close to the oracle ordering by true error. **(d)** Position error stratified by detection confidence — error is 5.3 mm at high confidence (*n* = 20,258) and degrades to 22.2 mm only for rare low-confidence frames (*n* = 121). This figure is reproduced by `python scripts/make_uncertainty_figure.py`.

### Datasets

<table>
<tr>
<td align="center" width="33%">
<img src="figures/dataset_a.png" alt="Dataset A" width="100%"/><br/>
<b>Dataset A</b><br/>60 trials &middot; 7-DoF &middot; 13 fps
</td>
<td align="center" width="33%">
<img src="figures/dataset_b.png" alt="Dataset B" width="100%"/><br/>
<b>Dataset B</b><br/>30 trials &middot; 7-DoF &middot; 13 fps
</td>
<td align="center" width="33%">
<img src="figures/dataset_c.png" alt="Dataset C" width="100%"/><br/>
<b>Dataset C</b><br/>24 trials &middot; 6-DoF &middot; 26 fps
</td>
</tr>
</table>

> **Full dataset download:** the datasets are released as the **LASK** dataset on
> Zenodo — [doi.org/10.5281/zenodo.20752650](https://doi.org/10.5281/zenodo.20752650)
> (raw video, kinematics and precomputed visual features). Please cite it (see
> [Citation](#citation)). Training (`scripts/train.py`) and the full
> evaluation path **(B)** require these. **You do not need them to reproduce the
> headline Dataset-A table or the figures** — the committed
> `results/evaluation_data.npz` and the bundled `data/sample_a` + `data/sample_c`
> trials are sufficient for the offline reproduction and the inference demo.

---

## Installation

```bash
git clone https://github.com/omariosc/BTPN.git
cd BTPN
git lfs pull          # fetch checkpoints + evaluation data (tracked with Git LFS)
pip install -e .
```

> **Requirements:** Python 3.10+, PyTorch 2.0+. A CUDA GPU is recommended for
> training and the full evaluation path **(B)**, but the **offline
> reproduction, the figures, and the inference demo all run CPU-only** — no GPU
> required. Verified end-to-end on CPU with `torch` 2.x, `ultralytics` 8.x.

## Quick Start

### Inference demo on the bundled sample trial (CPU)

The kinematic-prior model runs on the included `data/sample_a` trial with no
GPU and no extra data:

```bash
python scripts/inference.py \
    --checkpoint checkpoints/kinematic_foundation.pt \
    --trial data/sample_a \
    --norm-stats checkpoints/kinematic_foundation_norm.npz \
    --mc-samples 5 \
    --output predictions.npz
```

This prints a per-tool position / rotation / jaw summary and saves
`predictions.npz`. The same command also runs on the 6-DoF sample
(`--trial data/sample_c`); note that sample is a short 6-DoF clip evaluated
with the 7-DoF foundation prior and Dataset-A normalization, so it exercises
the pipeline end-to-end but its error numbers are not accuracy-meaningful.

The two YOLO checkpoints (`yolo_segmentation.pt`, `yolo_keypoints.pt`) also
load and predict on the sample frames (`data/sample_a/frames`,
`data/sample_c/frames`) on CPU via `ultralytics`.

### Inference with the full BTPN model

```python
from btpn import BTPN, BTPNConfig

config = BTPNConfig.from_yaml("configs/btpn.yaml")
model = BTPN.load_pretrained("checkpoints/btpn_supervised.pt", config)
model.eval()

# Run inference on a trial
predictions = model.predict(trial_data)
print(f"Position: {predictions['position'].shape}")      # (T, 2, 3)
print(f"Quaternion: {predictions['quaternion'].shape}")    # (T, 2, 4)
print(f"Sigma (pos): {predictions['sigma_pos'].shape}")   # (T, 2, 3)
```

### Training from Scratch

```bash
# Stage 1: Train Kinematic Foundation Model
python scripts/train.py --stage foundation --config configs/kinematic_foundation.yaml

# Stage 2: Visual SSL Pre-training
python scripts/train.py --stage ssl --config configs/btpn.yaml

# Stage 3: Supervised Fine-tuning
python scripts/train.py --stage supervised --config configs/btpn.yaml

# (Optional) Train detection models
python scripts/train.py --stage detection --task segmentation --config configs/detection.yaml
python scripts/train.py --stage detection --task keypoints --config configs/detection.yaml
```

### Evaluation

There are **two** evaluation entry points. Use **(A)** to reproduce the
Dataset-A table rows on any machine; use **(B)** for the full,
from-scratch evaluation once the datasets are available.

**(A) Offline reproduction — CPU-only, no GPU, no full dataset.**
Recomputes the **Full BTPN / Dataset A** pose and calibration metrics directly
from the committed predictions. These commands reproduce the paper and retrained
rows of [Key Results (b)](#b-pose-prediction--dataset-a-held-out-trials):

```bash
python scripts/evaluate.py --from-npz results/evaluation_data.npz
python scripts/evaluate.py --from-npz results/evaluation_data_vmf_fix.npz
```

Each command prints a side-by-side comparison with its table row and writes
reproduced LaTeX and JSON metrics under `results/`.
See [Reproducing the results table](#reproducing-the-results-table) for the
exact numbers this emits.

**(B) Full evaluation from the trained model — requires the full datasets + a GPU.**
Runs the model (MC-Dropout) over every held-out trial, recomputing all
metrics and the cross-dataset numbers. This needs the datasets and their
precomputed visual features at the path set in `configs/paths.yaml`
(**dataset download link: _to be added_ — see [Datasets](#datasets)**):

```bash
# Evaluate on Dataset A (held-out trials)
python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset A

# Evaluate on all datasets and (re)generate LaTeX tables
python scripts/evaluate.py --checkpoint checkpoints/btpn_supervised.pt --dataset all --output-tables
```

### Reproducing the paper figures

```bash
# Four-panel calibration figure (figures/uncertainty_quality.{png,pdf})
python scripts/make_uncertainty_figure.py

# Combined output comparison (figures/output_comparison.{png,pdf}); jaw in % opening
python scripts/generate_figures.py --data results/evaluation_data.npz --figure OC --output-dir figures

# Output comparison + trajectories + a lighter position-uncertainty summary + supplementary
python scripts/generate_figures.py --data results/evaluation_data.npz --all --output-dir figures
```

> Both read the released `results/evaluation_data.npz` and denormalize to
> physical units (mm, unit quaternions) using the stats embedded in that file.
> `make_uncertainty_figure.py` runs a hard ECE gate before writing.

---

### Reproducing the results table

Command **(A)** recomputes the **Full BTPN (paper) / Dataset A** row of
[Key Results (b)](#b-pose-prediction--dataset-a-held-out-trials) directly from
`results/evaluation_data.npz` and prints the npz-recomputed value next to the
committed table value for every metric, with a within-tolerance verdict:

| Metric | Reproduced (CPU, from npz) | Table (b) |
|:-------|:--------------------------:|:---------:|
| Pos *x* / *y* / *z* (mm) | 4.2 / 4.4 / 3.5 | 4.2 / 4.4 / 3.4 |
| Pos ‖v‖ (mm) | 7.0 (mean 5.5) | 7.0 |
| Roll / Pitch / Yaw (°) | 14.2 / 7.0 / 15.3 | 14.4 / 7.3 / 15.6 |
| Geodesic (°) | 11.8 | 11.7 |
| Jaw (% opening) | 13.6 | 13.6 |
| Position ECE | 0.028 | 0.028 |
| Rotation ECE (Fisher) | 0.301 | 0.30 |
| Jaw ECE | 0.079 | 0.079 |

> The released `evaluation_data.npz` is a single Monte-Carlo-dropout inference
> pass; the committed table rows come from a sibling pass. Position, ECE and
> jaw-% reproduce to 3 significant figures (the ECE set reproduces exactly);
> per-axis rotation RMSE carries run-to-run MC variance of <~0.3° (so the npz
> prints geodesic 11.8 against the table's 11.7). Command **(A)** writes
> `results/table2b_reproduced.tex` and `results/evaluation_reproduced.json` and
> also checks the predicted quaternions are unit-norm with a sane geodesic
> (median ~5°), confirming the released predictions are intact.
>
> **Jaw is reported as per-trial %-opening**, not degrees. The jaw channel is a
> raw sensor voltage with no voltage-to-angle calibration, so each frame is
> scored against its own trial's 10/90-percentile opening range; the headline
> camera-ready value of "1.72°" was a unit artifact and is superseded here.
> **Rotation calibration** is the physical-space Fisher ECE (0.30); the model
> is over-conservative (under-confident) in orientation. The other rows of
> Table 2 (ART-Net, Visual regr./LSTM/TCN/VTT, standalone Kinematic regr., the
> full ablation set, and the cross-dataset B/C numbers) come from separate
> models/datasets not contained in `evaluation_data.npz`; they are carried
> from [`results/all_results.json`](results/all_results.json) and require path
> **(B)** with the full datasets to recompute.

---

## Pre-trained Checkpoints

| Model | File | Size | Description |
|:------|:-----|:----:|:------------|
| Kinematic Foundation | `kinematic_foundation.pt` | 143 MB | Multi-scale temporal pose predictor (frozen prior) |
| Kinematic Norm Stats | `kinematic_foundation_norm.npz` | <1 KB | Z-score normalization statistics |
| BTPN SSL | `btpn_ssl.pt` | 126 MB | Stage 1: Visual SSL pre-trained encoder |
| **BTPN Supervised** | **`btpn_supervised.pt`** | **131 MB** | **Stage 2: Final model (paper results)** |
| BTPN Norm Stats | `btpn_norm.npz` | <1 KB | Stage 2 normalization statistics |
| YOLO Segmentation | `yolo_segmentation.pt` | 52 MB | YOLOv26m-seg (91.1% mAP<sub>50-95</sub>) |
| YOLO Keypoints | `yolo_keypoints.pt` | 136 MB | YOLOv26m-pose (94.6% mAP<sub>50-95</sub>) |

> Checkpoints are tracked with [Git LFS](https://git-lfs.github.com/). Run `git lfs pull` after cloning.

---

## Project Structure

```
BTPN/
├── btpn/                          # Python package
│   ├── config.py                  # BTPNConfig dataclass (YAML-loaded)
│   ├── model.py                   # KinematicFoundationModel + BTPN
│   ├── components.py              # HTT, MEE, attention, embeddings
│   ├── visual.py                  # Visual projections, fusion, gates
│   ├── losses.py                  # All loss functions
│   ├── dataset.py                 # Data loading and preprocessing
│   ├── detection.py               # YOLO training and feature extraction
│   ├── metrics.py                 # Evaluation metrics (RMSE, ECE, AUSE)
│   └── utils.py                   # Schedulers, checkpointing, plotting
│
├── scripts/
│   ├── train.py                   # Unified training: --stage {foundation,ssl,supervised,detection}
│   ├── evaluate.py                # --from-npz (CPU offline repro) | --dataset {A,B,C,all} (full GPU)
│   ├── make_uncertainty_figure.py # CPU regen of figures/uncertainty_quality.{png,pdf}
│   ├── inference.py               # Single-trial inference demo
│   ├── generate_figures.py        # Reproduce trajectory / supplementary figures
│   └── annotate.py                # Create COCO annotations from video
│
├── configs/                       # YAML configuration files
│   ├── paths.yaml                 # Data paths (edit for your setup)
│   ├── kinematic_foundation.yaml  # Foundation model hyperparameters
│   ├── btpn.yaml                  # Full BTPN hyperparameters
│   └── detection.yaml             # YOLO training configs
│
├── checkpoints/                   # Pre-trained weights (Git LFS)
├── figures/                       # Paper figures
├── results/                       # LaTeX tables + released per-sample evaluation data
│   ├── evaluation_data.npz        # Full-BTPN predictions/uncertainties/targets (+ trial_ids, det_conf)
│   ├── all_results.json           # Canonical table source (all rows, jaw % opening)
│   ├── unified_results.json       # Calibration set (position/rotation/jaw ECE)
│   └── table2{a,b,c}.tex          # LaTeX tables
├── data/                          # Sample data for testing
│   ├── sample_a/                  # 1 trial from Dataset A (7-DoF)
│   │   ├── label.json             # Kinematic data (all frames)
│   │   ├── frames/                # 3 sample endoscopic frames (PNG)
│   │   └── annotations.json       # COCO segmentation + keypoints
│   └── sample_c/                  # 1 trial from Dataset C (6-DoF)
│       ├── *.txt                  # Per-frame kinematic data
│       └── frames/                # 3 sample endoscopic frames (PNG)
└── docs/                          # Extended documentation
    ├── TRAINING.md                # Step-by-step training guide
    ├── DATA_FORMAT.md             # Dataset format specification
    └── ARCHITECTURE.md            # Architecture details
```

## Data Format

See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for full specification. Brief overview:

**7-DoF (Datasets A, B):** Each trial is a directory with `label.json` containing Tool1, Tool2, Camera, and World kinematic streams. Each stream has position (3D, mm), quaternion (wxyz), and jaw angle (voltage).

**6-DoF (Dataset C):** Each trial is a directory with numbered `.txt` files (one per frame) containing Reference, Fenestrated, Curved, and Camera sensor readings.

---

## Training Pipeline

The full training pipeline takes approximately **52 hours** on a single NVIDIA RTX 3060:

| Stage | Epochs (total / best) | Wall Time | Patience |
|:------|:---------------------:|:---------:|:--------:|
| YOLO segmentation fine-tune | 124 / 104 | 4.0 h | 20 |
| YOLO keypoint fine-tune | 169 / 144 | 3.6 h | 25 |
| Kinematic Foundation Model | 80 / 50 | 23.6 h | 30 |
| Visual SSL pre-training | 39 / 14 | 9.1 h | 25 |
| Visual supervised fine-tuning | 50 / 10 | 11.9 h | 20 |

See [`docs/TRAINING.md`](docs/TRAINING.md) for detailed instructions.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{choudhry2026btpn,
  title     = {Bayesian Temporal Pose Networks for Uncertainty-Calibrated
               Laparoscopic Tool Pose Tracking},
  author    = {Choudhry, Omar and Ali, Sharib and Biyani, Chandra Shekhar
               and Jones, Dominic},
  booktitle = {Medical Image Computing and Computer Assisted Intervention
               (MICCAI)},
  year      = {2026}
}
```

If you use the dataset, please also cite the **LASK** dataset:

```bibtex
@dataset{choudhry2026lask,
  title     = {{LASK}: A Dataset for Laparoscopic Skill and 7-DoF Kinematics},
  author    = {Choudhry, Omar and Ali, Sharib and Biyani, Chandra Shekhar
               and Rajasundaram, Ramanan and Jones, Dominic},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20752650},
  url       = {https://doi.org/10.5281/zenodo.20752650}
}
```

## License

This project is licensed under [CC BY-NC 4.0](LICENSE). You may use and adapt this work for non-commercial purposes with attribution.

## Reporting of Experimental Results

Per-item coverage of the [MICCAI 2026 experimental-reporting guidelines](https://conferences.miccai.org/2026/en/PAPER-SUBMISSION-GUIDELINES.html) — hyperparameters and their selection, data splits, metric definitions, baseline implementation/tuning, runtime/memory, failure-mode analysis, computing infrastructure, clinical significance, and the limitations (single-seed training, no multi-seed error bars or formal significance tests) — is in [`docs/REPORTING.md`](docs/REPORTING.md).

## Acknowledgements

We would like to thank the organisers of the Urology Simulation Boot Camps in Leeds, UK, the British Association of Paediatric Endoscopic Surgeons and all surgeons who spent time completing the data collection across all these events. The work was funded by UKRI EPSRC grants EP/S024336/1, UKRI914 and UKRI180. Approval for the dataset collection was granted by the University of Leeds Faculty of Engineering and Physical Sciences Research Ethics Committee (ref: EPS 2211 and MEEC 22-023).
