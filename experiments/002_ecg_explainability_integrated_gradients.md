# Experiment 002: ECG Explainability with Integrated Gradients

**Date:** 5 September 2026
**Run date:** 25 August 2026
**Status:** Completed initial explainability evaluation
**Repository:** [`Cardiac-Nexus-Lab/nexus-ai-engine`](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine)
**Notebook:** [`03_ecg_explainability.ipynb`](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine/blob/main/notebooks/03_ecg_explainability.ipynb)

## Objective

Attach a feature-attribution method to the trained ECG baseline so that each prediction is accompanied by an inspectable explanation rather than a probability alone. The objective of this experiment is to establish that attribution can be computed reliably for the baseline model and rendered against the source waveform in a form a clinician could review.

This experiment addresses the explainability requirement of the project directly. It does not attempt to validate the clinical correctness of the explanations.

## Method

Integrated Gradients (Sundararajan, Taly and Yan, 2017) attributes a model output to each input feature by integrating the gradient of the output along a straight-line path from a baseline input to the actual input:

```text
IG_i(x) = (x_i - x'_i) × ∫[α=0 to 1] ∂F(x' + α(x - x')) / ∂x_i dα
```

For a 12-lead ECG tensor of shape 12 × 1,000, the method produces one attribution value for each of the 12,000 input samples. The integral is approximated by a Riemann sum over a fixed number of interpolation steps.

The method was selected over Grad-CAM because it does not require a convolutional final layer. This keeps the approach valid if the ECG encoder is later replaced by a recurrent, transformer, or state-space architecture.

## Model under analysis

The attribution was computed against a reproduction of the Experiment 001 baseline, retrained on 25 August 2026 using the same configuration and random seed.

| Metric | Experiment 001 | Reproduction run |
| --- | ---: | ---: |
| Test loss | 0.536877 | 0.536946 |
| AUROC | 0.920406 | 0.920736 |
| Average precision | 0.812641 | 0.813770 |
| F1-score | 0.725581 | 0.721839 |
| Sensitivity | 0.850909 | 0.856364 |
| Specificity | 0.834951 | 0.827670 |

Reproduction confusion matrix:

|  | Predicted non-MI | Predicted MI |
| --- | ---: | ---: |
| Actual non-MI | 1,364 | 284 |
| Actual MI | 79 | 471 |

The two runs agree to within approximately 0.0004 AUROC. The small differences are consistent with nondeterminism between execution environments; Experiment 001 was trained on a Colab GPU runtime and the reproduction was trained on a Colab CPU runtime after GPU capacity became unavailable mid-session. The reproduction is therefore treated as confirming the baseline rather than superseding it.

## Attribution configuration

- Library: Captum `IntegratedGradients`
- Attributed quantity: predicted probability, obtained by wrapping the model output in a sigmoid
- Baseline input: zero signal across all 12 leads, representing an absence of electrical activity
- Interpolation steps: 64
- Data source: PTB-XL fold 10 only, the held-out test split
- Examples selected: two MI-labelled and two non-MI-labelled recordings

Attributing the probability rather than the raw logit was chosen so that attribution magnitudes correspond to the quantity presented to a reader.

## Results

Attribution completed successfully for all four selected recordings.

| Example | True label | Predicted P(MI) |
| --- | --- | ---: |
| 1 | MI | 0.852 |
| 2 | MI | 0.660 |
| 3 | Non-MI | 0.047 |
| 4 | Non-MI | 0.039 |

Each result was rendered as a 12-panel figure showing the standardized waveform for each lead in black, with attribution shaded beneath it: positive attribution, indicating support for the MI prediction, in red; negative attribution, indicating support against it, in blue. Attribution was normalized against the maximum absolute attribution across all leads of the recording so that relative importance between leads remains visible.

One figure has been committed as a reference artifact:

```text
nexus-ai-engine/results/explainability/mi_example_1_p0.852.png
```

The remaining three figures were reviewed during the session but were not retained as files. The notebook reproduces all four.

## Qualitative observations

In the two MI-labelled examples, attribution was visibly concentrated around the QRS complexes and the segment immediately following them, which is the region of the cardiac cycle where ischemic changes are conventionally assessed. In the two non-MI examples, attribution was comparatively flat and diffuse with no sustained region of positive support.

This contrast is consistent with the model responding to localized morphology rather than to global signal properties or recording artifacts. It is a qualitative observation from four examples and is not a validation result.

## Limitations

- Four examples are not a sample from which any general claim about attribution quality can be made.
- No clinician has reviewed whether the highlighted regions correspond to the diagnostic features that justified each recording's original label.
- Attribution describes the behavior of this specific trained model. It is not evidence that the model has learned clinically valid features, and a model can attend to a correct region for an incorrect reason.
- Attribution was not compared against alternative methods such as Grad-CAM, DeepLIFT, or occlusion, so method-specific artifacts cannot be excluded.
- The convergence delta reported by Captum was printed during the run but was not recorded, so the numerical reliability of each approximation is not documented here. Future runs should record it.
- The underlying model remains a binary MI classifier with the limitations already recorded in Experiment 001.

## Reproduction

Run [`01_first_ecg_model.ipynb`](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine/blob/main/notebooks/01_first_ecg_model.ipynb) first so that the dataset and trained checkpoint are present in the runtime, then run [`03_ecg_explainability.ipynb`](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine/blob/main/notebooks/03_ecg_explainability.ipynb) in the same session. The explainability notebook loads the checkpoint, rebuilds the test split, computes attribution, and renders the figures.

Alternatively, the committed checkpoint may be loaded directly:

```text
nexus-ai-engine/results/checkpoints/cardio_nexus_ecg_mi_baseline.pt
```

## Change to model-artifact policy

Experiment 001 recorded that the trained checkpoint would be kept outside version control until a model-artifact storage policy was defined. That position has been superseded. The baseline checkpoint, 245 KB, is now committed to `nexus-ai-engine` so that recorded metrics can be verified without retraining.

The policy now in effect is that small checkpoints, meaning those under approximately 1 MB, may be committed directly alongside their metric record. Larger artifacts produced by future multimodal work will require external storage or Git LFS and a revised policy.

## Next experiment considerations

- Record the convergence delta for every attributed example.
- Retain all generated figures as artifacts rather than a single reference example.
- Extend attribution to the multi-label model once it has been trained, attributing each diagnostic superclass independently.
- Arrange clinical review of a small set of attribution figures to assess whether highlighted regions align with the reasoning of a human reader.
