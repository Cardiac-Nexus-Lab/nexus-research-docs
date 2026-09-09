# Cardiac MRI track: scoping document

Status: proposal. No MRI code or data exists in `nexus-ai-engine` at the time of writing.
Written against the conventions already established by the ECG track (`src/cardiac_nexus/models.py`,
`training.py`, `evaluate.py`, `explain.py`).

## 0. What I measured before writing this

> **Editor's note, added on filing.** The 2.1 GiB free-disk figure measured in
> section 0 was accurate when this was written, during a period when the machine
> was at 100% capacity. Roughly 5 GB of regenerable caches were cleared shortly
> afterwards, leaving about 6.3 GB free. The measurement is left as written
> because the conclusions it drives — fetch ED/ES only, never the 4D cine — remain
> correct and the headroom is not guaranteed to persist.


Three numbers in this document come from measurements taken on the target machine or from
file listings, not from memory. They matter enough to state up front.

**Disk is tighter than assumed.** `df -h /` reports **2.1 GiB available**, not ~9 GB. Any plan that
assumes several gigabytes of headroom is already wrong. This single fact eliminates more options
than the memory ceiling does.

**The MPS allocation ceiling is 5.73 GB.** `torch.mps.recommended_max_memory()` returns
5.73e9 bytes on this machine (PyTorch 2.13.0, macOS 26.5.2). That is the working figure for
"how much can a model actually use", and it is shared with everything else the system is doing.

**3D operations work on MPS.** I ran a minimal op-support probe (tiny tensors, a few seconds):
`Conv3d` forward and backward, `ConvTranspose3d`, `InstanceNorm3d`, `BatchNorm3d`, `MaxPool3d`
and trilinear `interpolate` all execute on the MPS device without CPU fallback. Public discussion
of PyTorch MPS still describes `conv_transpose3d` as unsupported
([pytorch/pytorch#130256](https://github.com/pytorch/pytorch/issues/130256)); on this torch version
that is out of date. This changes the conclusion in section 3 substantially, and it is the reason
that section does not simply say "3D is impossible here".

---

## 1. Dataset survey

### 1.1 Comparison

Sizes marked **measured** were obtained by reading file listings (HuggingFace tree API, or the
publisher's own download page) without transferring the data. Sizes marked *unverified* are ones I
could not source and am not going to invent.

| Dataset | Subjects | Labels | Size | Licence | Access | Verdict on this machine |
| --- | --- | --- | --- | --- | --- | --- |
| **ACDC** | 150 (100 train / 50 test) | Seg masks (LV cavity, myocardium, RV) at ED + ES; **and** 5 diagnostic classes, 30 patients each | **78 MB** for ED/ES frames + masks, all 150 patients (measured); +1023 MB for the full 4D cine (measured) | Not stated on the challenge page; cite Bernard et al. 2018 | Free account on the humanheart-project portal; download is a direct link once registered. Days at most. | **Fits easily.** The 78 MB working set is the whole point. |
| **Sunnybrook (SCD)** | 45 | LV endo/epi contours at ED + ES; 4 pathology groups | **2.70 GB** DICOM across 5 batches (433.6 + 839.2 + 594.2 + 469.3 + 363.4 MB, measured from the Cardiac Atlas Project download page); contours 1.37 MB | **CC0 1.0 (public domain)** | Direct download, no registration | Marginal. 2.70 GB against 2.1 GB free. DICOM would need converting and the originals deleting. Doable but awkward. |
| **M&Ms** | 375 (150 annotated train + 25 unannotated; 40 val; 160 test) | Seg masks (LV, MYO, RV) at ED + ES; vendor and centre labels; pathology labels | *unverified*. Cine 4D, 192×192 to 384×384, 5–16 slices, 18–30 frames → my estimate is several GB for the full cine, but I could not source a figure. An ED/ES-only subset would be roughly ACDC-scale. | *unverified* | Registration with the organisers; a third-party listing points at a MEGA folder. The official ub.edu site was unreachable when I checked. Wait time unknown, plausibly weeks. | Conditional. ED/ES-only would fit; full cine probably would not. |
| **M&Ms-2** | 360 (160 / 40 / 160) | RV-focused seg masks, short-axis **and** long-axis; 7 pathologies + healthy | *unverified* | *unverified* | Same portal; unreachable at time of writing | Same conditional verdict. |
| **EMIDEC** | 150 (100 train / 50 test) | DE-MRI masks: cavity, normal myocardium, **infarct**, no-reflow; plus clinical variables; normal vs pathological label | *unverified*; short-axis DE-MRI covering the LV only, so my estimate is well under 1 GB | **CC BY-NC-SA 4.0** | Registration on emidec.com required | Very likely fits. Different contrast mechanism (late gadolinium), so it does not compose with ACDC. |
| **UK Biobank** | ~500k enrolled; 61,292 with both ECG and CMR at the imaging visit; 44,463 12-lead resting ECGs (field 20205) | Cine CMR, tagging, T1 mapping, aortic flow; derived phenotypes; linked outcomes; **paired ECG** | Tens of TB for bulk imaging | Managed access, application-specific terms | Free researcher registration reviewed within ~10 working days, then a **paid** application (tiered; I could not verify current fee figures). Historical application-to-release time was ~24 weeks as of 2018. | **Fails outright.** Not the memory — the disk, the money, and the months. |

### 1.2 The ACDC size finding, in detail

This is the load-bearing measurement. I enumerated a public mirror of ACDC
(`viennh2012/cardiac_cine_acdc` on HuggingFace, 758 files) via the metadata API and summed by
filename suffix:

| File type | Count | Total | Median |
| --- | ---: | ---: | ---: |
| `*_sax_ed.nii.gz` (end-diastole image) | 150 | 38.5 MB | 0.26 MB |
| `*_sax_ed_gt.nii.gz` (ED mask) | 150 | 0.5 MB | ~0 MB |
| `*_sax_es.nii.gz` (end-systole image) | 150 | 38.2 MB | 0.26 MB |
| `*_sax_es_gt.nii.gz` (ES mask) | 150 | 0.4 MB | ~0 MB |
| `*_sax_t.nii.gz` (full 4D cine) | 150 | **1023.2 MB** | 7.16 MB |

**The labelled part of ACDC is 78 MB. The unlabelled 4D cine is 93% of the bytes.** Every
segmentation and diagnosis result in the ACDC literature is computed on ED and ES frames only. So
the dataset the project actually needs is smaller than the PTB-XL cache already on disk (~0.5 GB
per the README). It fits in RAM as a single array: 192×192×10×2×100 as `uint8` is **74 MB**, or
295 MB as float32 — which means `MRIDataset` can hold everything in memory and slice in
`__getitem__`, exactly like `ECGDataset` in `training.py` already does.

Two caveats on that mirror, stated because they change what the numbers mean. First, it is a
third-party copy, not the official distribution; the project should download from the
humanheart-project portal and re-measure. Second, it is **preprocessed**: I read one NIfTI header
directly and found `patient001` at 192×192×10, datatype uint8, voxel spacing 1.0 × 1.0 × 10.0 mm.
The original ACDC has in-plane resolution of 1.37–1.68 mm²/pixel per the challenge documentation,
so this mirror has been resampled to 1 mm and intensity-quantised. The official data will be
somewhat larger and will need its own resampling step. Treat 78 MB as a lower bound of the right
order, not as the exact figure for the official archive.

### 1.3 Access notes worth acting on early

ACDC and EMIDEC both require an account but no committee review, so registration can be done on
day one and is not on the critical path. M&Ms and M&Ms-2 involve the organisers and an unknown
wait; if they are wanted at all, apply early and treat them as a bonus rather than a dependency.
UK Biobank should be applied for only if someone is prepared to fund and wait for it, and nothing
in the plan below should depend on it.

---

## 2. Recommended task

**Recommendation: staged. Segmentation first, clinical measures derived from it second, diagnosis
classification third and framed as a demonstration rather than a result. On ACDC.**

### 2.1 Why segmentation is the primary task

Segmentation is the only task on this dataset with enough supervision to support an honest number.
Each ACDC patient contributes roughly ten annotated short-axis slices at two phases, so 100 training
patients yield on the order of 2,000 labelled 2D images and, more importantly, on the order of
10⁷ labelled pixels. Dice is estimated over pixels, so the confidence intervals are tight even
though the patient count is small. The ECG track's existing bootstrap machinery in `evaluate.py`
transfers directly, resampled at the **patient** level rather than the slice level, since slices
within a patient are anything but independent.

There is also a well-populated leaderboard to sit against, which is how the ECG track already frames
its results (README compares against Strodthoff et al. 2021). The ACDC challenge leaderboard reports
best-in-class Dice of 0.967 (ED) and 0.928 (ES) for the LV cavity, 0.946 / 0.904 for the RV, and
0.896 / 0.919 for the myocardium, attributed to Isensee and to Simantiris. A local 2D U-Net that
lands in the high 0.8s / low 0.9s is a legitimate, reportable result and does not require pretending
to be state of the art.

### 2.2 The overfitting question on the diagnosis task, answered directly

You asked me to address this rather than route around it, so: **the ACDC diagnosis task cannot
produce a defensible headline number on this dataset, and the project should not try to make it
one.**

The arithmetic is unforgiving. 100 training patients across five classes is 20 patients per class.
The test set is 50 patients, ten per class. On 50 cases, a single flipped prediction moves accuracy
by two points. Computing Wilson 95% intervals on the published figures makes the problem visible:

| Reported test accuracy | Wilson 95% CI |
| --- | --- |
| 1.00 (50/50) — Khened et al. | 0.929 – 1.000 |
| 0.92 (46/50) — Isensee et al. | 0.812 – 0.968 |
| 0.86 (43/50) — Wolterink et al. | 0.738 – 0.930 |
| 0.80 (40/50), hypothetical | 0.670 – 0.888 |

(Accuracy figures as reported in secondary summaries of the ACDC benchmark comparison; I did not
succeed in extracting the corresponding table from the Bernard et al. PDF directly, so treat the
attributions as needing confirmation against the primary source. Khened et al. state 100% in their
own abstract, which I did verify.)

The top three published methods have overlapping intervals. The metric is saturated and the test set
cannot separate them. Any new number the project produces on those 50 patients is measuring the seed
as much as the method.

There is a second, more interesting point. The methods that won the ACDC diagnosis task did not
classify images end-to-end. Khened et al. state explicitly that they extracted "clinically relevant
cardiac parameters and hand-crafted features" from their segmentation and trained an ensemble
classifier on those. The diagnosis task on ACDC is, in practice, a small-tabular problem downstream
of a good segmentation. That is not a limitation to work around; it is the correct shape of the
problem and it happens to be the shape that is most defensible on 100 patients — a handful of
physiologically meaningful features and a low-capacity classifier, rather than a deep network with
millions of parameters fitted to 100 examples.

So the honest framing is: **segmentation is the result; ejection fraction and the other derived
measures are the validation that the segmentation is clinically meaningful; classification is a
demonstration that the encoder/head architecture extends to a diagnostic head, reported with
intervals and explicitly labelled as underpowered.** That framing is consistent with how the ECG
README already treats its weakest class (the HYP paragraph) — state the weakness in the same
sentence as the number.

### 2.3 Why the intermediate stage matters most

Deriving end-diastolic volume, end-systolic volume, ejection fraction and myocardial mass from the
predicted masks is the highest-value, lowest-risk piece of the whole track, and it is often skipped.
It converts a Dice score, which no clinician has intuition for, into a quantity with a normal range,
a known measurement error, and an existing clinical decision threshold. It gives a second,
independent evaluation axis: the ACDC leaderboard reports EF correlation and bias alongside Dice,
so there is something to compare against. And it produces a model output that is *intrinsically*
interpretable, which matters a great deal for section 4.

### 2.4 Datasets rejected, and why

**Sunnybrook** is LV-only, 45 cases, and 2.70 GB of DICOM against 2.1 GB of free disk. Its CC0
licence is genuinely attractive and it is the right choice if licensing ever becomes a blocker, but
it is strictly less informative than ACDC and harder to store.

**EMIDEC** is a good dataset and a natural *second* target, because infarct segmentation on DE-MRI
connects directly to the ECG track's MI class. It is rejected as the first target only because
late-gadolinium contrast is a different imaging problem from cine, and starting there means the
encoder cannot be reused for the cine work later. Revisit it after ACDC.

**M&Ms / M&Ms-2** are the correct external validation sets for an ACDC-trained segmentation model —
that is literally what they were built for, and the M&Ms paper's own conclusion is about
generalisation to unseen vendors. They are rejected as a starting point purely on access
uncertainty. Keep them as the stage-5 external validation, which also closes the README's standing
"no external dataset has been tested" limitation for the MRI track from the outset.

**UK Biobank** is discussed in section 5.

---

## 3. Recommended architecture

### 3.1 The encoder/head contract

`models.py` defines the contract: an encoder maps a modality to a 128-dimensional embedding
(`EMBEDDING_DIM = 128`), and a task head maps that embedding to a prediction. Segmentation strains
this contract in one specific way, and it should be resolved deliberately rather than quietly.

A U-Net decoder needs the encoder's intermediate feature maps (skip connections), not just the
bottleneck vector. A fusion model needs only the vector. Both can be satisfied without breaking
`ECGEncoder`'s signature:

```
MRIEncoder.forward(x) -> Tensor            # [B, 128]  -- same contract as ECGEncoder
MRIEncoder.forward_features(x) -> (Tensor, list[Tensor])   # embedding + skips
```

`MRISegmenter` calls `forward_features` and attaches a decoder head; `MRIDiagnosisHead` and any
future `FusionHead` call `forward` and never see the skips. The comment at the top of `models.py`
already promises that "MRI and tabular encoders follow the same contract" — this keeps that promise
literally true for the fusion path while letting the segmentation path reach inside.

The proposed shape, following the module docstring style already used in the repo:

- **`MRISliceEncoder`** — a 2D convolutional encoder over one short-axis slice (or a small stack of
  adjacent slices as input channels; see 3.3). Returns a per-slice embedding plus skips.
- **`MRIEncoder`** — runs the slice encoder over a volume's slices and pools the per-slice
  embeddings into one 128-d volume embedding. Pooling should be masked mean plus a small attention
  pooling, because slice counts vary between patients (5–16 in M&Ms) and because the basal and
  apical slices are the ones that carry the disease signal. The attention weights are also, for free,
  a slice-level saliency that is cheap and honest — see 4.4.
- **`MRISegmenter`** — encoder + U-Net decoder head, trained per-slice.
- **`MRIDiagnosisClassifier`** — encoder + linear head, mirroring `ECGClassifier` exactly.

### 3.2 Memory: 2D vs 2.5D vs 3D

These are **my estimates**, computed from an explicit model, not measured. The model: sum the
conv output feature maps that autograd must retain for the backward pass, for a 4-level U-Net with
two convolutions per level, base width 32, doubling per level, encoder and decoder, float32; then
double the total as a rough allowance for gradients and the backward pass's own temporaries. It
ignores parameters and optimiser state, which for a U-Net of this size are tens of megabytes and
not the binding term. It will be wrong by a factor of well under two, which is enough to make the
comparison. The input shape 192×192 is the measured ACDC slice size from 1.2; the 16-slice depth is
padded from ACDC's typical ~10.

| Approach | Input | Batch | Stored activations (est.) | Peak with backward (est.) | Against the 5.73 GB MPS ceiling |
| --- | --- | ---: | ---: | ---: | --- |
| 2D slice-wise | 192×192 | 8 | 0.27 GB | ~0.55 GB | Comfortable |
| 2D slice-wise | 192×192 | 16 | 0.55 GB | ~1.09 GB | Comfortable |
| 2D slice-wise | 192×192 | 32 | 1.10 GB | ~2.19 GB | Fine |
| 2.5D (3 slices → channels) | 192×192×3ch | 16 | ≈ 2D + input term | ≈ 2D | Comfortable — the extra input channels cost almost nothing |
| Full 3D | 192×192×16 | 1 | 0.40 GB | ~0.80 GB | Fits |
| Full 3D | 192×192×16 | 2 | 0.80 GB | ~1.59 GB | Fits |
| Full 3D | 192×192×16 | 4 | 1.59 GB | ~3.19 GB | Fits, but little headroom while the machine does anything else |

### 3.3 The honest conclusion about 3D

**3D is not blocked by memory on this machine. It is blocked by data and by anisotropy.** I want to
be precise about this because the easy answer — "8 GB, therefore no 3D" — is wrong here, and the
real reasons are more useful.

The reason memory is not the blocker is that ACDC volumes are tiny. A short-axis cine stack is about
ten slices. A 192×192×16 volume is only 16× a single slice, and my estimate puts a batch-2 3D U-Net
at roughly 1.6 GB peak, comfortably inside the measured 5.73 GB. The MPS op-support probe in
section 0 confirms the required 3D operations run on-device. A 3D U-Net on ACDC would train here.

The reasons not to are these. **First, sample count.** Training 2D gives ~2,000 training images;
training 3D gives 200 (100 patients × 2 phases). The whole difficulty of this dataset is that 100
patients is not many, and going 3D multiplies that difficulty by ten in exchange for through-plane
context. **Second, anisotropy.** The measured voxel spacing on the mirror is 1.0 × 1.0 × 10.0 mm — a
ten-to-one ratio. An isotropic 3×3×3 kernel spans 3 mm in-plane and 30 mm through-plane; it is
modelling a physically absurd neighbourhood. nnU-Net (Isensee et al., *Nature Methods* 18:203–211,
2021) handles exactly this by configuring anisotropic kernels and pooling per dataset, and its ACDC
configuration is the reason its results are strong; reproducing that heuristic by hand is real work
with no guaranteed payoff. **Third, batch statistics.** Batch 1–2 makes BatchNorm unusable; the fix
(InstanceNorm or GroupNorm, which is what nnU-Net does) is fine but is another deviation from the
ECG track's conventions. **Fourth, wall-clock.** A 3D forward/backward at batch 2 does roughly the
work of a 2D pass at batch 32 but sees one tenth as many patients per epoch.

**What is given up by not going 3D:** through-plane context, which matters most at the apex and base
where slices are partial and the RV is hardest to delineate. The M&Ms-2 literature is explicit that
the RV base is where segmentation fails, and a 2D model has no way to know that the slice above it
was already outside the ventricle. It also gives up the ability to enforce 3D shape consistency, so
predicted masks can be inconsistent between adjacent slices in ways a human would never produce —
which shows up directly as noise in the derived volumes and therefore in ejection fraction.

**The mitigation, and the actual recommendation: 2.5D.** Feed each slice together with its immediate
neighbours (2 or 4 of them) as extra input channels, and predict the mask for the centre slice only.
This recovers most of the local through-plane context, keeps the training-sample count at ~2,000,
keeps batch sizes large enough for BatchNorm, and costs essentially nothing in memory — the extra
channels affect only the first convolution's input. It is a well-established compromise in cardiac
segmentation and it is the right default here.

So: **2.5D U-Net, 2 or 4 neighbour channels, batch 16, base width 32.** Run a plain 2D variant
(1 input channel) as the control so the neighbour channels' contribution is measurable rather than
assumed. If, and only if, the 2.5D model's derived ejection-fraction error is dominated by
slice-to-slice inconsistency, revisit 3D at batch 2 with anisotropic kernels — as a stage-6
experiment, not as the plan.

### 3.4 Things to carry over from the ECG track

Patient-level splits, always — leaking slices from one patient across the train/test boundary would
inflate Dice enormously and is the single easiest way to produce a fraudulent number here. The
`TrainConfig` dataclass pattern, the `results/<run_name>/` output convention with a metrics JSON, the
bootstrap intervals from `evaluate.py`, and the seed handling all transfer unchanged. Augmentation
needs a new module (`augment_mri.py`) — rotation, scaling, elastic deformation, intensity/gamma
shifts; the M&Ms challenge conclusion specifically identifies intensity-driven augmentation as what
distinguished generalising methods from non-generalising ones, so this is not boilerplate.

---

## 4. Explainability

### 4.1 The problem, stated plainly

Integrated Gradients attributes a **scalar** output back to the input. A classifier gives you that
scalar for free: the probability of a class. A segmentation network does not. It emits a logit per
voxel per class — on a 192×192 slice with 4 classes, 147,456 scalars. "Integrated Gradients on the
segmentation model" is therefore not a defined operation until you say which scalar you meant, and
whatever you choose *is* the explanation's question. Running the existing `explain.py` against a
segmentation head and shipping the resulting picture would be exactly the kind of thing the module's
own docstring warns against.

Worse, the obvious choices are close to vacuous. Attribute the logit at voxel *(i,j)* and you will
learn that the model looked at the neighbourhood of *(i,j)* — the receptive field, rendered as a
heatmap. That is a fact about the architecture, not about what the model learned.

### 4.2 Option A (recommended primary): make the explained object the derived clinical measure

The strongest explainability story available here does not involve a saliency map at all. If the
pipeline is `image → mask → volumes → ejection fraction`, then the model's output is already
decomposed into quantities a cardiologist reads directly. "This patient is classified DCM because
the predicted end-diastolic volume is X mL and the ejection fraction is Y%, and here is the
segmentation those were computed from, overlaid on the image" is a complete, faithful, checkable
explanation. It is faithful in the strict sense that the stated reasons are the actual computation,
not a post-hoc approximation of it — which is precisely what a saliency map is not.

This should be the headline. It also costs almost nothing to build once section 2.3 is done, and it
is the honest reading of why hand-crafted-feature methods won the ACDC diagnosis task.

### 4.3 Option B (recommended secondary): attribute the diagnosis head, reusing `explain.py` intact

The staged architecture in 3.1 hands this over for free. `MRIDiagnosisClassifier` is an encoder plus
a linear head producing five class logits — structurally identical to `ECGClassifier`. Integrated
Gradients against a class probability is then exactly the operation `explain.py` already implements,
including `_ProbabilityWrapper` and the convergence-delta check. The input is 2D or 3D instead of
1D, so the plotting changes and the baseline choice needs re-thinking (a zero image is not
"absence of signal" the way a zero ECG is; a blurred-image or dataset-mean baseline is more
defensible and should be stated), but the method and its sanity check port directly.

This gives the project a genuine like-for-like statement: the same attribution method, with the same
sanity check, applied to two modalities.

### 4.4 Option C: Grad-CAM and Seg-Grad-CAM, with the caveat attached

Grad-CAM (Selvaraju et al., ICCV 2017) on the encoder's final block gives a coarse spatial map and
is cheap. For the segmentation head, Seg-Grad-CAM (Vinogradova et al., AAAI 2020 student abstract)
is the established adaptation: replace the scalar class score with a **sum over a chosen set of
pixels**, then proceed as normal. It is defensible precisely because it forces you to declare the
set. Useful framings on ACDC: sum over all voxels predicted as myocardium, and ask which image
regions supported that; or sum over the voxels where the prediction disagreed with the ground truth,
and ask what drove the error. The second is more interesting and is closer to a debugging tool than
a trust-building one, which is the right register for this project.

The caveat that must ship with it: the choice of pixel set determines the map, so a Seg-Grad-CAM
figure without a stated aggregation region is uninterpretable.

The slice-attention weights from `MRIEncoder` (3.1) are a fourth, near-free signal: which slices the
volume embedding actually weighted. Attention weights are not explanations in the strong sense — the
attention-is-not-explanation literature is clear on that — and should be presented as a description
of the pooling, nothing more.

### 4.5 Option D: uncertainty maps, which are the most defensible of all

Per-voxel predictive uncertainty — Monte Carlo dropout (Gal and Ghahramani, ICML 2016), a small deep
ensemble (Lakshminarayanan et al., NeurIPS 2017), or test-time augmentation — has a property no
saliency method has: **it makes a falsifiable prediction.** If the uncertainty map is meaningful,
high-uncertainty voxels should be where the model is actually wrong. That is directly measurable
against the ground-truth masks. You can plot per-case mean uncertainty against per-case Dice and
report the correlation, or sweep a rejection threshold and show error dropping as coverage falls.

This fits the project's existing character better than anything else in this section. The ECG track
already invested in calibration (`evaluate.py`, temperature scaling after Guo et al. 2017) because
it cared whether the numbers meant anything. Uncertainty maps are the spatial version of the same
commitment, and unlike attribution they can be scored rather than admired. Test-time augmentation is
also the cheapest option computationally — no architectural change, no retraining — which matters
here.

### 4.6 Sanity checks, analogous to the model-randomization test

The ECG track's standard is that no attribution ships without a check. Four checks transfer or adapt:

**Cascading model randomization (Adebayo et al., NeurIPS 2018)** — already implemented in
`explain.py` for the ECG classifier. Port it unchanged to the MRI diagnosis head (Option B). For
Grad-CAM and Seg-Grad-CAM, run the same procedure: randomise weights layer by layer from the output
backwards and measure the rank correlation between the original map and each randomised map. A map
that survives is describing the input. This is the direct analogue and it is the one to lead with.

**Attribution localisation against the ground-truth mask** — this is an MRI-specific check with no
ECG analogue, and it is the most interesting thing available on this track. Because ACDC provides
pixel-level anatomy, you can measure what fraction of total positive attribution mass falls inside
the myocardium, the LV cavity, or the RV, versus outside the heart entirely. A DCM classification
whose attribution mass sits in the chest wall is not trustworthy regardless of how confident it is.
This is an adaptation of the "pointing game" family of localisation metrics from the natural-image
saliency literature (Zhang et al., IJCV 2018), applied with anatomical rather than object
annotations. It gives a single number per case, so it can be reported with a bootstrap interval like
everything else in the repo, and it can be compared across attribution methods — which directly
addresses the "methods disagree with one another" finding that `explain.py`'s docstring already
cites from Bender et al.

**Uncertainty-versus-error correlation** — described in 4.5. Report Spearman correlation between
per-case mean predictive entropy and per-case Dice, with an interval.

**Segmentation quality control without ground truth** — Reverse Classification Accuracy
(Valindria et al., IEEE TMI 2017), applied to cardiac MR at scale by Robinson et al.
(*Journal of Cardiovascular Magnetic Resonance*, 2019, on UK Biobank), predicts per-case
segmentation quality when no reference mask exists. This is not an explainability method, but it is
the thing that makes an explainability claim safe at deployment: it detects the failed segmentations
before anyone tries to explain them. Worth noting as future work rather than committing to in the
first pass.

---

## 5. The fusion blocker

### 5.1 A correction to the premise

The brief states that no public dataset provides ECG and MRI for the same patients. That is very
nearly true and is the right operating assumption, but it is not literally true, and the exception
matters. **UK Biobank acquires a resting 12-lead ECG and a cardiac MR scan at the same imaging
visit.** Published work using it reports 61,292 participants with both ECG and CMR available at the
imaging visit, and the 12-lead resting ECG field (20205) holds 44,463 recordings. Studies predicting
CMR-derived measures from the ECG in that cohort already exist — for instance, work on predicting
left ventricular hypertrophy from the 12-lead ECG in the UK Biobank imaging study
(*European Heart Journal – Digital Health* / PMC10393938, 2023).

So the paired cohort exists. It is simply **out of reach for this project**: a paid, tiered
application; a historical application-to-release time around 24 weeks; and bulk imaging measured in
terabytes against 2.1 GB of free disk. The right way to state this in the README is not "no such
data exists" — that is falsifiable and someone will falsify it — but "paired ECG–CMR data exists in
UK Biobank and is inaccessible to this project on cost, access time, and storage grounds."

### 5.2 What follows for the multimodal ambition

Three options, in descending order of what I would recommend.

**Be explicit that fusion is demonstrated architecturally, not validated clinically.** The encoder/
head split in `models.py` is a genuine engineering claim: two independently trained encoders emit
128-dimensional embeddings that a fusion head can consume without either being rewritten. That claim
can be *demonstrated* — build the fusion head, show it runs, show the shapes compose, show the
encoders load from their separate checkpoints. It cannot be *evaluated*, because there is no paired
cohort to evaluate on. Saying so is a stronger position than any workaround, and it is consistent
with how the README already handles the digitization caveat ("trained on rendered printouts, not
photographs of real ones. Performance on genuine clinical paper is untested"). The engineering is
real; the clinical claim is not made.

**Construct a synthetic paired cohort and label it as such.** ACDC and PTB-XL both carry diagnostic
labels, and some concepts overlap (ACDC's MINF against PTB-XL's MI; ACDC's HCM against PTB-XL's
HYP). One can pair records across the two datasets by matching labels, producing a "cohort" on which
a fusion head trains and reports a number. This is worth building **only as a plumbing test** and
must be labelled unambiguously as synthetic pairing. The number it produces is meaningless — the two
modalities are from different people, so any apparent fusion benefit is an artefact of the label
correlation you constructed. If it is done, the metric to report is "the pipeline executes
end-to-end", not AUROC.

**Apply to UK Biobank as a long-horizon item.** Free researcher registration takes about ten working
days and costs nothing; it can be started now regardless. The paid application should only follow if
someone has decided to fund it and can supply tens of TB of storage. Nothing in stages 1–6 should
depend on this.

There is a fourth possibility worth flagging as a research direction rather than a plan: the ECG
track's derived measures and the MRI track's derived measures meet at the same clinical quantities.
An ECG model that predicts reduced ejection fraction and an MRI model that measures ejection
fraction are estimating the same number from different signals. That gives a way to relate the two
tracks — a shared target — without needing paired patients. It is not fusion, and it should not be
called fusion, but it is a real connection and it is available.

---

## 6. Staged plan

Time estimates assume part-time work by one person on the described hardware. They are my estimates,
based on the scope of each stage rather than on any measurement, and the training times assume the
machine is not simultaneously running an ECG job. Every stage produces a committed artefact.

| Stage | Work | Estimate | Verifiable output |
| --- | --- | --- | --- |
| **1. Acquisition and inspection** | Register for ACDC (and EMIDEC, opportunistically). `src/cardiac_nexus/mri_data.py` following `data.py`'s resumable-download style, but fetching **only** the ED/ES frames and masks — never the 4D cine. Resample to a fixed in-plane spacing, centre-crop, cache as one array. Notebook: slice montages, per-class counts, voxel-spacing distribution, mask sanity. | 3–5 days | A ~100 MB cache built by one command, a `.gitignore` entry, and a short data-report notebook. Verifiable by re-running the fetch on a clean checkout. |
| **2. Segmentation baseline** | `MRISliceEncoder` / `MRIEncoder` / `MRISegmenter` in `models.py` honouring the 128-d contract. `augment_mri.py`. Training loop reusing `TrainConfig` and the `_run_epoch` structure. Dice + Hausdorff, patient-level bootstrap intervals via `evaluate.py`. Plain-2D control against 2.5D. | 1.5–2.5 weeks | `results/mri_seg_baseline/` with a metrics JSON and overlay figures. Comparable to the ACDC leaderboard numbers in section 2.1. |
| **3. Derived clinical measures** | EDV, ESV, EF, myocardial mass from predicted masks using the recorded voxel spacing. Bland–Altman and correlation against the same measures computed from ground-truth masks. | 3–5 days | An EF agreement plot and a bias/limits-of-agreement table. This is the stage that makes the segmentation mean something, and it is cheap. |
| **4. XAI, with checks** | Uncertainty via test-time augmentation first (cheapest, most defensible). Uncertainty-versus-Dice correlation. Seg-Grad-CAM with a declared aggregation region. Attribution-localisation metric against the ground-truth masks. | 1.5–2 weeks | `results/mri_explainability/` with figures **and** a JSON of check outcomes, mirroring `results/explainability_local/attribution_report.json`. A check that fails is a publishable result here, not a setback. |
| **5. External validation** | If M&Ms access arrived: evaluate the stage-2 model unchanged on M&Ms ED/ES. If not: hold-out by ACDC scanner field strength (1.5 T vs 3.0 T) as a weak internal proxy, clearly labelled as such. | 4–6 days, gated on access | A cross-dataset Dice table. Closes the "no external validation" limitation for the MRI track from day one, which the ECG track still carries. |
| **6. Diagnosis head and architectural fusion demo** | `MRIDiagnosisClassifier` on the 128-d embedding, plus a features-from-segmentation classifier as the comparison that the ACDC winners used. Report both with Wilson intervals on 50 test cases and the underpowering stated in the same paragraph. Port `explain.py`'s IG and model-randomization test to the diagnosis head. Then a `FusionHead` taking an ECG embedding and an MRI embedding, demonstrated to compose, explicitly not evaluated. | 1.5–2 weeks | `results/mri_diagnosis/` with intervals; a passing model-randomization check on the MRI head; a fusion shape test in `scripts/test_pipeline.py`. |

Total: roughly **7–10 weeks part-time** to the end of stage 6, with stages 1–3 (about 3–4 weeks)
delivering the substantive result on their own.

The ordering is deliberate. Stage 3 is placed before all the explainability work because a derived
ejection fraction is both a validation of stage 2 and the strongest interpretability artefact the
track will produce; if the project stopped after stage 3 it would still have something real. Stage 6
is last because it is the stage most likely to produce a number that cannot be defended, and it
should be built on top of a segmentation result that can be.

---

## 7. Summary of what fails on this hardware

Stated bluntly, as requested.

- **UK Biobank fails.** Terabytes, money, months. Not a candidate.
- **Full 4D cine training fails on disk, not on compute.** 1023 MB for ACDC's cine alone, against
  2.1 GB free, before any cache or checkpoints. Fetch ED/ES only. Any motion or temporal-dynamics
  modelling is out of scope for that reason, and that is a genuine loss — the ACDC classes include
  ones where wall motion is the discriminating feature.
- **Sunnybrook at 2.70 GB does not fit alongside the existing PTB-XL cache.** It could be made to
  fit with conversion and deletion, but it buys nothing ACDC does not already provide.
- **nnU-Net as a framework is not the right tool here.** Its results are excellent and it is the
  correct citation for how to handle anisotropy, but its full pipeline (extensive preprocessing,
  five-fold cross-validation, ensembling, large patch sizes) is built for a very different compute
  budget. Borrow its design decisions; do not run it.
- **Full 3D does not fail on memory** — this is the one place where the expected answer is wrong.
  It fails on training-set size (200 volumes versus 2,000 slices) and on 10:1 voxel anisotropy.
  2.5D is the reduced version that works, and it gives up through-plane consistency, most visibly
  at the apex and base and therefore in the derived ventricular volumes.
- **A defensible ACDC diagnosis result is not achievable at all**, on any hardware, because the test
  set is 50 patients and the leaderboard is saturated. This is a dataset limit, not a machine limit,
  and the plan handles it by reporting the diagnosis task with intervals and an explicit statement of
  underpowering rather than as a headline.

---

## References

- Bernard, O., Lalande, A., Zotti, C., Cervenansky, F., et al. "Deep Learning Techniques for
  Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?"
  *IEEE Transactions on Medical Imaging* 37(11):2514–2525, 2018. (ACDC.)
- Campello, V. M., Gkontra, P., et al. "Multi-Centre, Multi-Vendor and Multi-Disease Cardiac
  Segmentation: The M&Ms Challenge." *IEEE Transactions on Medical Imaging* 40(12):3543–3554, 2021.
- Martín-Isla, C., et al. "Deep Learning Segmentation of the Right Ventricle in Cardiac MRI: The
  M&Ms Challenge." *IEEE Journal of Biomedical and Health Informatics*, 2023.
  doi:10.1109/JBHI.2023.3267857.
- Lalande, A., et al. "Emidec: A Database Usable for the Automatic Evaluation of Myocardial
  Infarction from Delayed-Enhancement Cardiac MRI." *Data* 5(4):89, 2020.
- Lalande, A., et al. "Deep learning methods for automatic evaluation of delayed enhancement-MRI.
  The results of the EMIDEC challenge." *Medical Image Analysis*, 2022.
- Radau, P., Lu, Y., Connelly, K., Paul, G., Dick, A. J., Wright, G. A. "Evaluation Framework for
  Algorithms Segmenting Short Axis Cardiac MRI." *The MIDAS Journal – Cardiac MR Left Ventricle
  Segmentation Challenge*, 2009. (Sunnybrook.)
- Petersen, S. E., et al. "Cardiovascular magnetic resonance imaging in the UK Biobank: a major
  international health research resource." *European Heart Journal – Cardiovascular Imaging*
  22(3):251–, 2021 (protocol description; the original protocol paper is Petersen et al., *JCMR*
  18:8, 2016). *I did not independently verify the 2016 citation's page numbers.*
- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., Maier-Hein, K. H. "nnU-Net: a
  self-configuring method for deep learning-based biomedical image segmentation."
  *Nature Methods* 18:203–211, 2021.
- Khened, M., Alex, V., Krishnamurthi, G. "Fully Convolutional Multi-scale Residual DenseNets for
  Cardiac Segmentation and Automated Cardiac Diagnosis using Ensemble of Classifiers."
  *Medical Image Analysis*, 2019 (arXiv:1801.05173). Reports 100% ACDC diagnosis accuracy.
- Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
  Localization." *ICCV* 2017.
- Vinogradova, K., Dibrov, A., Myers, G. "Towards Interpretable Semantic Segmentation via
  Gradient-Weighted Class Activation Mapping (Student Abstract)." *AAAI* 2020. (Seg-Grad-CAM.)
- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B. "Sanity Checks for
  Saliency Maps." *NeurIPS* 2018. (Already used by `explain.py`.)
- Zhang, J., Bargal, S. A., Lin, Z., Brandt, J., Shen, X., Sclaroff, S. "Top-down Neural Attention
  by Excitation Backprop." *International Journal of Computer Vision* 126:1084–1102, 2018.
  (Pointing game.)
- Gal, Y., Ghahramani, Z. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in
  Deep Learning." *ICML* 2016.
- Lakshminarayanan, B., Pritzel, A., Blundell, C. "Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles." *NeurIPS* 2017.
- Valindria, V. V., et al. "Reverse Classification Accuracy: Predicting Segmentation Performance in
  the Absence of Ground Truth." *IEEE Transactions on Medical Imaging* 36(8):1597–1606, 2017.
- Robinson, R., Valindria, V. V., et al. "Automated quality control in image segmentation:
  application to the UK Biobank cardiovascular magnetic resonance imaging study."
  *Journal of Cardiovascular Magnetic Resonance* 21:18, 2019.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. "On Calibration of Modern Neural Networks."
  *ICML* 2017. (Already used by `evaluate.py`.)
- Strodthoff, N., Wagner, P., Schaeffter, T., Samek, W. "Deep Learning for ECG Analysis: Benchmarks
  and Insights from PTB-XL." *IEEE Journal of Biomedical and Health Informatics* 25(5):1519–1528,
  2021. (Already the ECG track's reference point.)

**Unverified items, listed so they are not mistaken for established facts:** M&Ms, M&Ms-2 and
EMIDEC download sizes; M&Ms and M&Ms-2 licence terms and current access route (the ub.edu site was
unreachable when I checked); current UK Biobank application fees; the exact attribution of the
0.92 and 0.86 ACDC diagnosis accuracies to Isensee et al. and Wolterink et al. (taken from a
secondary summary, not from the primary benchmark table); and whether the official ACDC archive's
ED/ES footprint matches the 78 MB I measured on a preprocessed third-party mirror.
