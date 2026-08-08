# Experiment 001: ECG MI versus Non-MI Baseline

**Date:** 8 August 2026  
**Status:** Completed baseline experiment  
**Repository:** [`Cardiac-Nexus-Lab/nexus-ai-engine`](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine)  
**Notebook:** [`01_first_ecg_model.ipynb`](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine/blob/main/notebooks/01_first_ecg_model.ipynb)

## Objective

Establish a reproducible baseline for classifying 12-lead ECG recordings according to the presence or absence of a myocardial infarction (MI) diagnostic label.

## Dataset

The experiment used PTB-XL version 1.0.3 from PhysioNet.

- Total recordings: 21,799
- Non-MI recordings: 16,330
- MI recordings: 5,469
- Signal format: 12-lead ECG
- Recording duration: 10 seconds
- Working sampling rate: 100 Hz
- Source: [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)

An ECG was assigned the positive label when its diagnostic statements included the `MI` superclass. Recordings without that label were assigned to the non-MI category. The non-MI category contains other diagnoses and should not be interpreted as a healthy-control category.

## Data split

The recommended patient-wise PTB-XL folds were used:

- Folds 1–8: training
- Fold 9: validation
- Fold 10: testing

This arrangement keeps records from the same patient within one split and reduces the risk of train/test leakage.

## Model and training configuration

The baseline used a compact one-dimensional convolutional neural network designed for ECG time-series input.

- Input shape: 12 leads × 1,000 samples
- Convolutional blocks: 3
- Pooling: max pooling followed by adaptive global pooling
- Output: binary classification logit
- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: `64`
- Epochs: `10`
- Random seed: `42`
- Loss: weighted binary cross-entropy with logits
- Runtime: Google Colab GPU

Each ECG lead was standardized independently before being passed to the model.

## Test results

| Metric | Result |
| --- | ---: |
| Test loss | 0.536877 |
| AUROC | 0.920406 |
| Average precision | 0.812641 |
| F1-score | 0.725581 |
| Sensitivity | 0.850909 |
| Specificity | 0.834951 |
| Accuracy | 0.84 |

### Classification report

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Non-MI | 0.94 | 0.83 | 0.89 | 1,648 |
| MI | 0.63 | 0.85 | 0.73 | 550 |
| Overall accuracy | — | — | 0.84 | 2,198 |

### Confusion matrix

|  | Predicted non-MI | Predicted MI |
| --- | ---: | ---: |
| Actual non-MI | 1,376 | 272 |
| Actual MI | 82 | 468 |

## Preliminary interpretation

The baseline achieved an AUROC of approximately 0.92 on the held-out PTB-XL test fold. At the default probability threshold of 0.5, the model identified 468 of 550 MI-labelled recordings and missed 82. It also classified 1,376 of 1,648 non-MI recordings correctly.

The difference between MI precision and MI recall indicates that the model identifies many MI-labelled recordings but also produces a number of false-positive MI predictions. Threshold selection and calibration require separate evaluation.

## Limitations

- This is a single baseline configuration and has not been compared against alternative architectures or classical baselines.
- PTB-XL is not a Mysuru-specific cohort and does not establish performance on the intended local population.
- The non-MI category contains other diagnoses and is not equivalent to healthy subjects.
- No external validation, calibration study, subgroup analysis, or clinical review has been completed.
- The result must not be interpreted as evidence of clinical readiness or as a standalone basis for medical decisions.

## Reproduction

Open the linked notebook in Google Colab and run it from top to bottom. The notebook downloads the public dataset into temporary runtime storage, constructs the labels, trains the baseline, and produces the evaluation metrics and plots.

The trained checkpoint was saved locally from the Colab runtime as:

```text
cardio_nexus_ecg_mi_baseline.pt
```

The checkpoint is intentionally kept outside the Git repository until a model-artifact storage policy is defined.

## Next experiment considerations

Before adding additional modalities, the baseline should be reviewed for reproducibility, threshold behavior, calibration, and comparison with a simpler reference model. Future results should be recorded using the same structure.
