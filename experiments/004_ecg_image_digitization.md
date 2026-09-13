# Experiment 004: Reading an ECG Waveform Back From a Printed Page

**Date:** 9 September 2026 (recorded 13 September 2026)
**Status:** Completed for flat pages and scans; angled photographs unresolved
**Repository:** [`Cardiac-Nexus-Lab/nexus-ai-engine`](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine)
**Code:** `src/cardiac_nexus/ecg_image/`, run via `scripts/train_digitizer.py` and `scripts/test_pipeline.py`
**Model:** `souller/cardiac-nexus-ecg-digitizer` on Hugging Face (private)

## Objective

Most people who have an ECG hold it as paper, not as a digital recording. The
objective was to recover a 12-lead signal from an image of a printout well enough
that the Experiment 003 classifier receives input resembling its training data.

## Approach

### Training data is generated, not collected

No public dataset pairs ECG printouts with the signals that produced them in
volume. Pairs were therefore synthesised: PTB-XL recordings from training folds
1–8 were rendered as ECG paper (12×1 layout, 25 mm/s, 10 mm/mV, 4 px/mm, 16 mm
rows), with amplitude randomised between 0.2 and 0.45 per unit and lead labels
drawn on half of pages. Each strip was then degraded with perspective, rotation,
a lighting gradient, a soft shadow, a crease, blur, sensor noise and JPEG
compression. A recording is never seen twice with the same distortion.

The distortion returns its homography and the ground-truth trace is transformed
through it. An early version discarded the homography, so labels described where
the trace had been before distortion; this was caught visually, as ground-truth
points sitting beside the drawn line.

### The task is framed as per-column localisation

Each lead is cut into four 2.5-second windows, which at 4 px/mm is 250 px wide,
so one pixel column is one sample. Windows are resized to 96 × 256. For every
column the model predicts a probability distribution over the 96 rows. Its
expectation gives a sub-pixel trace position; its peak probability gives a
per-sample confidence.

| | |
|---|---|
| Model | `TraceLocalizer`: 2D CNN, width pooled early, per-column row head |
| Parameters | 202,649 |
| Target | Gaussian over rows, σ = 1.5 px |
| Loss | Cross-entropy plus 0.1 × L1 on the expected row |
| Optimiser | AdamW, learning rate 2e-3, weight decay 1e-4, cosine schedule |
| Sampling | 500 fresh training records per epoch (24,000 strips); 120 fixed validation records from fold 9 |
| Hardware | Apple M1, 8 GB, MPS; about 6 minutes per epoch |

## Engineering that determined whether training was possible at all

**First design: 157 seconds per batch, about 21 hours per epoch.** Two causes
were measured rather than assumed. Width was never downsampled, so almost all
compute convolved 128 channels across 512 columns. And the batch exceeded
comfortable memory on an 8 GB machine: the same computation took 7,282 ms at 48
strips against 317 ms at 12. Pooling width early and halving the base width took
a step from 618 ms to 95 ms and the model from 861k to 202k parameters.

**Rendering was repeated twelve times per record**, once per lead. Rendering the
page once and cropping every lead from it took generation from 46 ms to 9.6 ms
per strip.

**The first full run was lost to a full disk.** It reached 1.29 px at epoch 11
and crashed at epoch 16 of 20. The best weights were held only in memory until
the final line of the script, so fifteen epochs of work disappeared. The trainer
now writes the checkpoint whenever validation improves.

## Results

Second run: 16 epochs, best at epoch 15.

| Measurement | Value |
|---|---|
| Validation strip error | **1.28 px** mean absolute |
| Strip-level correlation with the true trace (15 validation records, 720 strips) | 0.948 |
| Predicted-to-true trace standard deviation | 0.98 |
| **End-to-end, flat rendered page** (20 held-out test records) | **0.940** mean waveform correlation |
| Ceiling imposed by the paper format (100 records) | 0.964 |
| End-to-end, simulated angled photograph (12 test records) | 0.243 mean, 0.147 median |

The ceiling is the correlation between each true signal and that signal clipped
to its row, which is the most any reader could recover once QRS peaks run past
the edge of their row. At the tested amplitude clipping costs 0.036.

On flat pages the result is 97.5% of what the format permits, and the twelve
leads are uniform, rows 0–5 scoring 0.939 against rows 6–11 at 0.940.

## The pipeline, not the model, was the problem

An end-to-end smoke test against a partially trained checkpoint returned a
waveform correlation of **−0.010**. That is not an undertrained model. Four
defects were found by bisecting stage by stage.

| Defect | Evidence | Effect |
|---|---|---|
| One margin fraction used for both axes of a 1032 × 800 page | The 16 px margin is 0.0200 of the height but 0.0155 of the width; cuts landed 16 px low and 25 px late, compressing time by 5% | Correlation near zero |
| 256 pixel columns read as 250 samples | Four windows gave 1024 columns for 1000 samples, each window starting six samples later than the last | Clean-page correlation +0.165; 0.536 after fixing, over 20 records |
| Page detection accepted any quadrilateral covering a quarter of the frame | One detection returned 325 × 998 for an 800 × 1032 page, discarding 60% of the leads while reporting success | Now requires half the frame and the layout's aspect ratio |
| A page already flat was perspective-warped anyway | 0.940 untouched, 0.844 after a one-pixel crop, **0.543** through the warp (12 records) | 0.534 → **0.940** once near-identity warps are skipped |

The most informative measurement was one that showed no change. Nine further
epochs took strip error from 1.40 px to 1.28 px while end-to-end correlation went
from 0.536 to 0.534. That flat result established that the bottleneck was outside
the model, and prevented further training of a model that was already adequate.

## Negative results

Before the warp was identified, correlation fell steadily from the top of the
page to the bottom. Four explanations were proposed and each was eliminated by
measurement.

| Hypothesis | Test | Outcome |
|---|---|---|
| Chest leads clip more | Share of samples outside their row, 300 records | 2.67% limb against 2.98% chest: uniform. Per-lead standardisation equalises amplitude by construction |
| Strip grid drifts down the page | Offset between strip centre and rendered baseline, every row | Worst 0.90 px, symmetric about the middle; cannot produce a monotonic decline |
| Chest-lead morphology is harder | Render the leads in reverse order, so V6 is on top | Pattern followed the row, not the lead: rows 0–5 0.633 against rows 6–11 0.461 on the final checkpoint |
| Each window's baseline is offset | Align every window's median before joining | Correlation fell from 0.940 to 0.890 |

The model was also checked for flattening towards the baseline, which would give
low pixel error and poor correlation. It does not: predicted trace spread is 0.98
of the true spread. The actual cause, warp resampling error that grows with
distance across the transform, was in a stage that had been assumed harmless.

## Limitations

- **Angled photographs do not work.** The model is trained on strips distorted
  individually, whereas a photograph warps the whole page. The synthetic angled
  photographs also have no background border, so page detection has nothing to
  find and correctly declines.
- **Synthetic data only.** Nothing here is evidence about genuine clinical paper,
  other printers, other ECG machines, handwriting or stains.
- **12×1 layout only.** A standard 3×4 printout carries only 2.5 s of most leads
  and cannot supply the 10 s the classifier expects.
- **Absolute millivolts are not recovered**, only waveform shape. The classifier
  was trained on per-lead standardised signals and does not use them.
- End-to-end figures come from 12 to 20 records without bootstrap intervals, and
  should be read as estimates of the right order rather than precise values.

## Reproduction

```bash
python scripts/train_digitizer.py --epochs 16 --records-per-epoch 500 --val-records 120
python scripts/test_pipeline.py --records 20
```

## Next experiment considerations

- Render the page onto a background surface before distortion, so page detection
  becomes solvable and the simulation resembles a phone photograph.
- Train with page-level warps if angled photographs are required.
- Test on real printouts photographed with a phone, with the recording known.
