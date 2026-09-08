# Experiment 003: Multi-Label Classification, Architecture Search, and Calibration

**Date:** 8 September 2026
**Status:** Completed
**Repository:** [`Cardiac-Nexus-Lab/nexus-ai-engine`](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine)
**Code:** `src/cardiac_nexus/`, run via `scripts/train_ecg.py` and `scripts/evaluate_ecg.py`

## Objective

Extend the binary MI classifier of Experiment 001 to the five PTB-XL diagnostic superclasses, and establish how close a locally trainable model can come to the published benchmark for this task.

A secondary objective was to report results honestly: with intervals that reflect what the test set can support, and with probabilities that mean what they claim.

## Change of compute

Training moved from Google Colab to local hardware, an Apple M1 with 8 GB of memory, using the PyTorch MPS backend.

| | Colab (CPU runtime) | M1 via MPS |
| --- | ---: | ---: |
| Seconds per epoch, baseline CNN | 159 | 12.6 |
| Fifteen epochs | ~40 min | ~2 min |

The measured difference was roughly twelvefold, and local runs are not subject to session limits, disconnection, or repeated dataset download. Every result below was produced on the laptop.

## Task and data

PTB-XL v1.0.3, 100 Hz, official patient-wise folds. Recordings carrying no diagnostic superclass were dropped, leaving 21,388: 17,084 training, 2,146 validation, 2,158 test.

Labels are multi-hot over five superclasses, so a recording may carry more than one. Class frequencies are uneven, and the loss applies per-class positive weighting from 1.25 for NORM to 7.06 for HYP.

| Class | Test positives |
| --- | ---: |
| NORM | 963 |
| MI | 550 |
| STTC | 521 |
| CD | 496 |
| HYP | 262 |

## Results across configurations

Every configuration selects its reported model by validation macro AUROC rather than taking the final epoch.

| Configuration | Val macro AUROC | Test macro AUROC |
| --- | ---: | ---: |
| CNN, 15 epochs | 0.9089 | 0.9048 |
| CNN, 40 epochs | 0.9089 | 0.9063 |
| xresnet1d18, 30 epochs | 0.9056 | stopped early |
| CNN + augmentation + weight decay, 40 epochs | 0.9117 | 0.9056 |
| **xresnet1d18 + augmentation + weight decay + one-cycle, 25 epochs** | **0.9130** | **0.9107** |

Published reference: xresnet1d101 reaches 0.928 on this task (Strodthoff et al., IEEE JBHI 2021).

### Best configuration, per class

| Class | AUROC (95% CI) | Average precision (95% CI) | Sensitivity | Specificity |
| --- | --- | --- | ---: | ---: |
| NORM | 0.941 (0.932–0.950) | 0.912 (0.894–0.930) | 0.911 | 0.811 |
| MI | 0.921 (0.907–0.933) | 0.822 (0.791–0.850) | 0.776 | 0.891 |
| STTC | 0.930 (0.918–0.942) | 0.814 (0.778–0.848) | 0.852 | 0.867 |
| CD | 0.924 (0.909–0.938) | 0.843 (0.815–0.872) | 0.859 | 0.847 |
| HYP | 0.837 (0.813–0.860) | 0.474 (0.415–0.539) | 0.725 | 0.808 |

The MI figure of 0.921 matches the binary baseline of Experiment 001 to three decimal places, which is a useful check that moving to multi-label did not degrade the original task.

## What the negative results establish

Three interventions were tried individually and none improved test performance. They are recorded because they constrain what the remaining gap can be attributed to.

**Longer training does not help.** Validation macro AUROC reached 0.9089 at epoch 14 and returned exactly that value at epochs 21 and 33, while training loss fell from 0.487 to 0.410 and validation loss rose from 0.589 to 0.641. The apparent upward trend at epoch 14 was noise. The model converges and then overfits.

**More capacity alone makes it worse.** xresnet1d18 has 6.4 million parameters against the CNN's 59 thousand. Trained with the same flat learning rate it peaked at epoch 6 and declined for thirteen consecutive epochs, ending below the far smaller model.

**Augmentation alone does not transfer to test.** Adding augmentation and weight decay to the CNN moved validation from 0.9089 to 0.9117 and pushed the best epoch from 14 to 27, so it delays overfitting as intended. Test moved from 0.9063 to 0.9056, which is no change. HYP average precision improved from 0.467 to 0.493, the one class where it clearly helped.

**Together they work.** The same architecture that failed alone reaches 0.9107 when combined with augmentation, weight decay, and a one-cycle learning rate schedule, training productively through to epoch 20 rather than collapsing at epoch 6. Each ingredient was necessary and none was sufficient, which is why ablating one at a time made a working recipe appear useless.

The remaining 0.017 gap to the published figure is most plausibly attributable to depth. xresnet1d101 was measured at 682 seconds per epoch on this hardware, making a forty-epoch run over seven hours, so it was not attempted.

## Calibration

Discrimination and calibration are separate properties. A model can rank recordings correctly while its probabilities are systematically wrong, and only the latter matters if a reader interprets the number rather than the ordering.

Standard temperature scaling made calibration worse, from 0.096 to 0.102 mean expected calibration error. A single shared scalar assumes every class is over-confident by the same factor, which does not hold when training applied per-class weights ranging from 1.25 to 7.06.

Fitting a scale and intercept per class corrects it:

| Class | ECE raw | ECE calibrated |
| --- | ---: | ---: |
| NORM | 0.058 | 0.019 |
| MI | 0.053 | 0.017 |
| STTC | 0.078 | 0.010 |
| CD | 0.110 | 0.010 |
| HYP | 0.157 | 0.019 |
| **Mean** | **0.091** | **0.015** |

The calibrator is selected by validation ECE rather than assumed, and both candidate transforms are monotonic within a class, so AUROC and average precision are unchanged at 0.9107.

## Uncertainty

Per-class metrics carry 95% percentile bootstrap intervals over 1,000 resamples. The intervals are not decorative: HYP average precision is 0.474 with an interval from 0.415 to 0.554, spanning 0.14, because only 262 test recordings carry that label. Reporting the point estimate alone would imply a precision the data cannot support.

## Limitations

- Test performance sits 0.017 below the published benchmark, and the architecture most likely to close that gap is impractical on this hardware.
- HYP remains weak where it matters. AUROC of 0.837 sounds acceptable, but average precision of 0.474 is the honest figure for a rare class, and the model finds a minority of true cases at the default threshold.
- All results are from a single random seed. Seed-to-seed variation has not been quantified, so small differences between configurations should not be over-read.
- Evaluation remains confined to PTB-XL fold 10. No external dataset has been tested, so generalisation beyond this cohort, its recording equipment, and its labelling conventions is unestablished.
- The 0.5 decision threshold is a default, not a clinically chosen operating point.
- Nothing here establishes clinical safety or readiness for diagnostic use.

## Reproduction

```bash
python scripts/fetch_data.py
python scripts/train_ecg.py --arch xresnet1d18 --epochs 25 --augment \
    --weight-decay 1e-4 --one-cycle --learning-rate 3e-3 \
    --output results/local_xresnet18_full
python scripts/evaluate_ecg.py --checkpoint results/local_xresnet18_full/ecg_multilabel.pt
```

Artifacts, including the checkpoint and the metric records for every configuration above, are in the AI engine repository under `results/`.

## Next experiment considerations

- Quantify seed-to-seed variation before comparing configurations that differ by less than about 0.005.
- External validation, which is the largest outstanding gap in credibility. Chapman-Shaoxing and the Georgia dataset are public and use the same 12-lead format, but their labels are rhythm-oriented rather than diagnostic, so this requires label harmonisation and is a project rather than a download.
- Choose an operating threshold deliberately, per class, against a stated clinical preference between missed cases and false alarms.
