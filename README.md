# Person-Independent Eye Movement Classification with a Hybrid CNN+LSTM Model

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 1. Description

This repository contains the official codebase and implementation for the paper:
*"Person-Independent Eye Movement Classification with a Hybrid CNN+LSTM Model:
A Leave-One-Person-Out Evaluation on MPIIGaze".*

The objective of this study is to evaluate, under a strictly person-independent
protocol, which architectural components are needed to classify the three
fundamental eye-movement types — **fixation, saccade, and smooth pursuit** —
from short gaze-vector sequences. We compare a convolutional model (1D-CNN), a
recurrent model (LSTM), and a hybrid (1D-CNN+LSTM), and assess them with
**leave-one-person-out (LOPO) 15-fold cross-validation**, in which each
participant is held out entirely for testing in turn so that no individual
appears in both the training and test sets. The comparison is supported by
formal statistical testing (McNemar's test, Friedman's test, and Cohen's *d*
effect sizes).

## 2. Dataset Information

The model was trained and evaluated using the publicly available **MPIIGaze**
dataset.

- **Participants:** 15 subjects (p00–p14) recorded during unconstrained,
  in-the-wild laptop use.
- **Input:** per-frame 3D gaze vectors read from
  `Data/Original/pXX/dayYY/annotation.txt` (gaze columns 26–28). The raw
  eye-region images are not used.
- **Sequences:** 213,508 overlapping 10-frame windows, shape `(10, 3)`, at 30 Hz.
- **Labels:** assigned by thresholding the per-window standard deviation σ of
  the gaze magnitude (τ₁ = 37, τ₂ = 50; the 25th/75th percentiles):
  σ < 37 → Fixation, 37 ≤ σ ≤ 50 → Smooth Pursuit, σ > 50 → Saccade. σ is used
  **only** to generate the ground-truth labels and is never provided to the
  model as an input feature.
- **Evaluation:** leave-one-person-out at the participant level to strictly
  prevent cross-participant data leakage.

## 3. Repository Structure

The repository is structured to ensure end-to-end reproducibility:

```
.
├── lopo_experiment.py     Main experiment: data loading, sequence construction
│                          and labeling, training of all three models, the LOPO
│                          loop, and the statistical analysis. Writes per-fold
│                          results and pooled predictions to results/.
├── training_history.py    Trains a single representative fold (default p11) and
│                          records its 30-epoch training/validation curves.
├── visualize.py           Generates the publication figures from the files in
│                          results/. No values are hard-coded; every number is
│                          read from the experiment outputs.
├── requirements.txt       Python dependencies.
├── LICENSE                MIT license.
├── README.md
│
├── results/               Experiment outputs (read by visualize.py)
│   ├── lopo_fold_results.csv        Per-fold accuracies for the three models
│   ├── yt_all.npy                   Pooled ground-truth labels
│   ├── hyb_all.npy                  Pooled CNN+LSTM predictions
│   ├── cnn_all.npy                  Pooled CNN-only predictions
│   ├── lstm_all.npy                 Pooled LSTM-only predictions
│   ├── fig1_person_distribution.csv Per-participant class composition
│   ├── fig1_sigma_distribution.csv  Sampled σ values per class
│   ├── fig4_training_history.csv    Representative-fold (p11) training history
│   └── lopo_preds/                  Per-fold predictions (preds_pXX.json)
│
└── figures/               Generated figures (PNG/PDF/TIFF, 300 DPI)
```

## 4. Requirements

The project is built using Python 3.11. The exact versions used to produce the
reported results are listed in `requirements.txt`:

`tensorflow==2.19.0`, `numpy==2.0.2`, `pandas==2.3.3`, `scikit-learn==1.6.1`,
`scipy==1.16.3`, `statsmodels==0.14.6`, `matplotlib==3.10.0`.

## 5. Usage Instructions

Follow these steps to replicate the study:

**Step 1: Data Preparation**
Download the MPIIGaze dataset from its original source
(<https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild>)
and set the `ORIG` path at the top of `lopo_experiment.py` to its
`Data/Original` directory. The raw dataset is not redistributed here due to its
original license.

**Step 2: Environment Setup**
```
pip install -r requirements.txt
```

**Step 3: Run the Experiment** (global seed 42 is enforced)
```
python lopo_experiment.py
```
This trains CNN-only, LSTM-only, and CNN+LSTM for each held-out participant,
writing the per-fold results, pooled predictions, and statistics into
`results/`. Each fold is saved immediately, so the run can be resumed if
interrupted.

**Step 4: Training Curves and Figures**
```
python training_history.py     # representative-fold training history
python visualize.py            # generates the figures into figures/
```

`visualize.py` reads only from `results/`; if you use the provided result files,
the figures can be regenerated without re-running the full experiment.

## 6. Results

Main results under 15-fold LOPO cross-validation:

| Model      | Params  | Mean Accuracy (SD) | 95% CI            |
|------------|---------|--------------------|-------------------|
| CNN-only   | 182,083 | 82.45% (5.31)      | [79.76, 85.13]    |
| LSTM-only  | 125,699 | 83.42% (9.47)      | [78.63, 88.21]    |
| **CNN+LSTM** | 215,811 | **87.83% (5.24)**  | **[85.18, 90.48]** |

The hybrid model is both the most accurate and the most stable across
participants (McNemar χ² = 3676.60 vs CNN and 6831.96 vs LSTM, both *p* < 0.001;
Friedman χ² = 10.80, *p* = 0.0045; Cohen's *d* = 1.02 vs CNN, 0.58 vs LSTM).
Aggregated per-class F1-scores are 0.895 (Fixation), 0.860 (Saccade), and
0.889 (Smooth Pursuit).

## 7. Citation

If you use this codebase in your research, please cite our paper (currently
under review):

A. Akkaya, *"Person-Independent Eye Movement Classification with a Hybrid
CNN+LSTM Model: A Leave-One-Person-Out Evaluation on MPIIGaze"*, Submitted for
publication, 2026.

(Note: The full citation details will be updated upon the paper's acceptance and
publication.)

Please also cite the MPIIGaze dataset:

X. Zhang, Y. Sugano, M. Fritz, A. Bulling, *"MPIIGaze: Real-World Dataset and
Deep Appearance-Based Gaze Estimation"*, IEEE TPAMI, 2019, 41(1), 162–175.

## 8. License

This project is licensed under the MIT License — see the `LICENSE` file for
details. The MPIIGaze dataset is subject to its own separate license.
