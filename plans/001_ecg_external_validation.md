# External Validation of the Cardiac Nexus ECG Classifier

**Scope.** A plan for evaluating the trained `xresnet1d18` five-superclass model
(PTB-XL v1.0.3, 100 Hz, macro AUROC 0.9107 on fold 10) on a cohort it has never
seen. Written before any data is downloaded and before any inference is run, so
that the acceptance criteria are fixed in advance rather than chosen after seeing
the numbers.

**Status of every number below.** Dataset sizes, sampling rates, licences and
per-code record counts are taken from the sources cited inline. The per-code
counts in the mapping tables come from `dx_mapping_scored.csv` and
`dx_mapping_unscored.csv` in the PhysioNet/CinC Challenge 2021 evaluation
repository, which tabulates every SNOMED-CT code against every constituent
database. Where I could not verify something I have written that I could not
verify it. Nothing here is estimated unless it is explicitly labelled as an
arithmetic estimate from stated assumptions.

---

## 1. Executive summary, including the parts that are bad news

Three findings dominate everything else in this document.

**First, PTB-XL is itself a member of the PhysioNet/CinC Challenge 2020 and 2021
aggregated datasets.** The obvious move — download `challenge-2021`, run the
model over it, report the number — is not an external validation. It is partly an
evaluation on the training set. The 2021 bundle contains a `PTB_XL` source with
21,837 records and a separate `PTB` (PTB Diagnostic) source, plus 10,000 records
described only as "augmented undisclosed" whose provenance cannot be checked.
Any external cohort assembled from that bundle must exclude PTB-XL, should
exclude PTB Diagnostic on independence grounds discussed in §2.4, and must
exclude the undisclosed records. This is the single easiest way to manufacture an
impressive and meaningless result, and it is worth stating in the eventual
write-up that it was avoided deliberately.

**Second, MI cannot be externally validated at scale on open 12-lead data with
these label sets.** This is a negative finding and it is not fixable by trying
harder. The two datasets everyone reaches for first are close to useless for MI:
the Georgia database carries the SNOMED code for myocardial infarction on **7**
records out of ~10,344, and Chapman-Shaoxing on **40** out of 10,646. Ningbo has
189 across all three MI codes combined. The Shandong Provincial Hospital database
has roughly 270. Only CPSC-Extra, at 3,453 records, has a substantial MI
population (~1,600 across `MI`, `old MI`, `anterior MI`, `chronic myocardial
ischemia`). MI is the second-strongest class in the current model (0.921) and it
is the class an external result would be most interesting for, so this is a
genuine disappointment rather than a technicality. Plan for MI to be reported
either from CPSC-Extra alone with wide intervals, or as "not externally
evaluated".

**Third, the largest external datasets label ECG *morphology findings*, whereas
PTB-XL superclasses are built from *diagnostic* statements, and PTB-XL is
explicit about the difference.** In PTB-XL's own `scp_statements.csv`, `TAB_`
(T-wave abnormality), `STD_` (non-specific ST depression), `STE_` (non-specific
ST elevation), `INVT` (inverted T-waves), `QWAVE` (Q waves present) and `VCLVH`
(voltage criteria for LVH) are **form** statements with `diagnostic = 0`. They
carry no superclass. Meanwhile `NST_`, `NDT`, `LNGQT` and `DIG` are both form and
diagnostic and do map to STTC. The high-count SNOMED codes in Chapman, Ningbo and
Georgia — `t wave abnormal` (11,716 total), `left ventricular high voltage`
(5,401), `st changes` (5,009), `t wave inversion` (3,989), `st depression`
(3,645), `qwave abnormal` (2,076) — are precisely the ones that correspond to
PTB-XL form statements. Sweeping them into STTC, HYP and MI would roughly triple
the external positive counts and would be exactly the "manufacture a result"
failure mode. Leaving them out is the consistent choice, and it is also the
choice that shrinks the usable external cohort the most.

**Recommendation.** Attempt the **Shandong Provincial Hospital (SPH) database**
first, and the **Georgia G12EC database** second. The reasoning is in §5. The
short version is that SPH is the only candidate whose label ontology can be
mapped onto PTB-XL superclasses without inventing semantics — in particular it is
the only one with an explicit "Normal ECG" statement, so NORM does not have to be
faked from "sinus rhythm and nothing else".

---

## 2. Candidate datasets

### 2.1 Comparison table

Sources: the PhysioNet project pages for `challenge-2021` v1.0.3,
`ecg-arrhythmia` v1.0.0 and `mimic-iv-ecg` v1.0; Zheng et al. (2020) for
Chapman-Shaoxing; Liu et al. (2022) for SPH. Record counts marked † are the
public-training-set counts tabulated in the Challenge 2021 `dx_mapping_*.csv`
files and differ from the totals quoted on the challenge landing page, which
appear to include hidden validation and test partitions.

| Dataset | Records | Rate | Length | Leads | Format | Label scheme | Licence | Access | Download |
|---|---:|---:|---|---|---|---|---|---|---|
| **SPH** (Shandong Provincial Hospital) | 25,770 | 500 Hz | 10–60 s (94% are 10–15 s) | 12 | HDF5 | AHA statements: 44 primary + 15 modifiers | CC BY 4.0 | Open, Figshare | Not stated on the sources I read |
| **Georgia G12EC** (Emory) | 10,344 | 500 Hz | 5–10 s | 12 | WFDB (MATLAB v4 `.mat` + `.hea`) | SNOMED-CT in `#Dx:` header lines | CC BY 4.0 | Open | Part of the 12.6 GB Challenge 2021 bundle; ~1.2 GB standalone (arithmetic estimate: 10,344 × 12 × 5,000 samples × 2 bytes) |
| **Chapman-Shaoxing** | 10,646 | 500 Hz | 10 s | 12 | WFDB | SNOMED-CT | CC BY 4.0 | Open | Part of `ecg-arrhythmia` (5.1 GB unzipped / 2.3 GB zip for Chapman + Ningbo together) |
| **Ningbo First Hospital** | 34,506 | 500 Hz | 10 s | 12 | WFDB | SNOMED-CT | CC BY 4.0 | Open | as above |
| **CPSC 2018** (main) | 6,877 | 500 Hz | 6–60 s | 12 | WFDB | 9 classes, SNOMED-mapped | CC BY 4.0 | Open | Part of the 12.6 GB bundle |
| **CPSC-Extra** | 3,453 | 500 Hz | 6–144 s | 12 | WFDB | SNOMED-CT | CC BY 4.0 | Open | Part of the 12.6 GB bundle |
| **PTB Diagnostic** | 516 | 1000 Hz | variable | 12 (+3 Frank) | WFDB | SNOMED-CT (challenge version) | CC BY 4.0 | Open | Part of the 12.6 GB bundle |
| **INCART (St Petersburg)** | 74 | 257 Hz | 30 min | 12 | WFDB | SNOMED-CT | CC BY 4.0 | Open | Part of the 12.6 GB bundle |
| **MIMIC-IV-ECG** | ~800,000 | 500 Hz | 10 s | 12 | WFDB | machine measurements + ~600k free-text cardiologist reports; **no structured diagnostic codes** | ODbL v1.0 | The v1.0 project page states open access | 33.8 GB zip / 90.4 GB unzipped |

Chapman-Shaoxing and Ningbo are distributed together as PhysioNet
`ecg-arrhythmia` v1.0.0, 45,152 records total. The 10,646-vs-45,152 confusion
appears widely in secondary sources; 10,646 is Chapman-Shaoxing alone, from
Zheng et al. (2020), and Ningbo is the 34,506-record remainder added later.

### 2.2 Credentialing

**Every dataset in the table above is open access.** None of them requires
PhysioNet credentialing. This removes what I had assumed would be the main
scheduling risk from the project, and it is worth saying so plainly in the
write-up rather than implying a bureaucratic obstacle that does not exist.

For completeness: PhysioNet's credentialed tier requires a CITI Program "Data or
Specimens Only Research" course plus a "Conflicts of Interest" module, an
affiliation questionnaire, and a reference. **I could not verify a typical
turnaround time** — the PhysioNet CITI course page documents the training
requirement but says nothing about review timelines, and I am not going to
guess. If a credentialed dataset later becomes necessary, treat the timeline as
unknown and start the application early.

The MIMIC-IV-ECG waveform module being listed as open access surprised me and I
would double-check it before relying on it; linkage to the MIMIC-IV clinical
tables is a separate, credentialed resource regardless.

### 2.3 Datasets that are unusable, and why

**INCART.** 74 records, 30 minutes each, 257 Hz. Too small for any interval you
would want to report, and the 30-minute continuous format is a different
recording regime from a 10-second resting ECG. Exclude.

**MIMIC-IV-ECG.** 800,000 records is tempting, but there are no structured
diagnostic labels — only machine measurements and free-text reports. Deriving
five superclasses from report text means building a clinical NLP labeller, and
then the external validation result becomes a joint test of the ECG model and an
unvalidated text classifier. That is a larger project than the one being scoped
and it would weaken rather than strengthen the credibility claim. Exclude for
now; revisit only if a published, validated report-to-superclass labeller for
MIMIC-IV-ECG appears.

**CPSC 2018 (main).** Only nine label classes, covering AF, first-degree AV
block, LBBB, RBBB, PAC, PVC, ST depression and ST elevation. Four of the nine are
rhythm findings that PTB-XL keeps outside the diagnostic hierarchy entirely, and
two more (STD, STE) fall into the form-statement trap described in §3. What
remains is essentially a CD-only evaluation. Not useless, but not a
five-superclass validation.

### 2.4 PTB Diagnostic: a specific independence concern

PTB Diagnostic has 368 MI records, which makes it the second-best public source
of MI labels. But PTB-XL was curated by the Physikalisch-Technische
Bundesanstalt, and PTB Diagnostic is the earlier PTB database from the same
institution. **I was not able to establish whether there is any patient overlap,
equipment overlap, or annotator overlap between the two.** Given that the entire
point of this exercise is a credible independence claim, using a dataset whose
independence I cannot demonstrate would undermine the result. Either resolve the
question with the dataset authors before using it, or exclude it and say why.

---

## 3. Label harmonisation

### 3.1 The structural problem

PTB-XL's five superclasses are defined by `scp_statements.csv`, which I read from
the local copy at `/Users/yashas/Documents/nexus-ai-engine/data/ptbxl/scp_statements.csv`.
Of 71 statements, only 44 have `diagnostic = 1` and therefore contribute a
superclass. The full membership is:

| Superclass | PTB-XL diagnostic statements |
|---|---|
| **NORM** (1) | `NORM` — "normal ECG" |
| **CD** (11) | `1AVB`, `2AVB`, `3AVB`, `CRBBB`, `IRBBB`, `CLBBB`, `ILBBB`, `LAFB`, `LPFB`, `IVCD`, `WPW` |
| **MI** (14) | `AMI`, `ASMI`, `ALMI`, `IMI`, `ILMI`, `IPMI`, `IPLMI`, `LMI`, `PMI`, `INJAS`, `INJAL`, `INJIN`, `INJIL`, `INJLA` |
| **STTC** (13) | `NDT`, `NST_`, `ISC_`, `ISCAN`, `ISCAS`, `ISCAL`, `ISCIN`, `ISCIL`, `ISCLA`, `DIG`, `LNGQT`, `EL`, `ANEUR` |
| **HYP** (5) | `LVH`, `RVH`, `LAO/LAE`, `RAO/RAE`, `SEHYP` |

The 27 non-diagnostic statements are 12 rhythm statements (`SR`, `AFIB`, `AFLT`,
`STACH`, `SBRAD`, `SARRH`, `SVARR`, `SVTAC`, `PSVT`, `PACE`, `BIGU`, `TRIGU`) and
15 form-only statements (`ABQRS`, `PVC`, `PAC`, `PRC(S)`, `STD_`, `STE_`, `TAB_`,
`NT_`, `INVT`, `LOWT`, `HVOLT`, `LVOLT`, `QWAVE`, `LPR`, `VCLVH`).

Three consequences follow, and they drive the whole mapping.

**Rhythm findings have no home.** Atrial fibrillation, flutter, sinus
bradycardia, tachycardia, premature complexes and pacing are *not* members of any
superclass. `src/cardiac_nexus/data.py` drops any record with no diagnostic
superclass. A PTB-XL record whose only finding is AF was therefore never shown to
the model. An external record whose only finding is AF must be dropped the same
way — not scored as five negatives.

**Form findings have no home either, and this is the expensive one.** `TAB_` and
`STD_` are excluded from STTC while `NST_` and `NDT` are included. `QWAVE` is
excluded from MI. `VCLVH` — "voltage criteria (QRS) for left ventricular
hypertrophy" — is excluded from HYP while `LVH` is included. PTB-XL is drawing a
line between a measurement and a diagnosis, and the SNOMED-coded datasets
frequently code only the measurement.

**Only one statement makes a record NORM.** PTB-XL's `NORM` is a positive
assertion by a human reader that the ECG is normal. It is not "sinus rhythm", and
it is not "we found nothing". Any external dataset without an explicit normality
statement forces NORM to be reconstructed from absence of findings, which makes
NORM prevalence a function of how thoroughly that source coded abnormalities.

### 3.2 Proposed mapping — Tier A, defensible without argument

These SNOMED codes have a direct counterpart among the 44 PTB-XL diagnostic
statements. Counts are from `dx_mapping_scored.csv` / `dx_mapping_unscored.csv`,
public training partitions.

| SNOMED | Name | → | PTB-XL counterpart | Georgia | Chapman | Ningbo | CPSC-Extra |
|---|---|---|---|---:|---:|---:|---:|
| 270492004 | 1st degree AV block | CD | `1AVB` | 769 | 247 | 893 | 106 |
| 195042002 | 2nd degree AV block | CD | `2AVB` | 23 | 8 | 58 | 21 |
| 27885002 | complete heart block | CD | `3AVB` | 8 | 1 | 75 | 27 |
| 233917008 | AV block (degree unspecified) | CD | `_AVB` subclass | 74 | 166 | 78 | 5 |
| 713427006 | complete RBBB | CD | `CRBBB` | 28 | 0 | 1096 | 113 |
| 59118001 | RBBB | CD | `CRBBB` | 542 | 454 | 195 | 1 |
| 713426002 | incomplete RBBB | CD | `IRBBB` | 407 | 0 | 246 | 86 |
| 733534002 | complete LBBB | CD | `CLBBB` | 0 | 0 | 213 | 0 |
| 164909002 | LBBB | CD | `CLBBB` | 231 | 205 | 35 | 38 |
| 251120003 | incomplete LBBB | CD | `ILBBB` | 86 | 0 | 6 | 42 |
| 6374002 | bundle branch block (unspec.) | CD | any BBB | 116 | 0 | 385 | 0 |
| 445118002 | left anterior fascicular block | CD | `LAFB` | 180 | 0 | 380 | 0 |
| 445211001 | left posterior fascicular block | CD | `LPFB` | 25 | 0 | 5 | 0 |
| 698252002 | nonspecific IVCD | CD | `IVCD` | 203 | 235 | 536 | 4 |
| 74390002 | WPW pattern | CD | `WPW` | 2 | 4 | 68 | 0 |
| 164865005 | myocardial infarction | MI | superclass | 7 | 40 | 83 | 376 |
| 54329005 | anterior MI | MI | `AMI` | 0 | 0 | 57 | 62 |
| 57054005 | acute MI | MI | MI statements | 0 | 0 | 49 | 0 |
| 164867002 | old MI | MI | MI statements | 0 | 0 | 0 | 1168 |
| 164861001 | myocardial ischemia | STTC | `ISC_` | 0 | 0 | 0 | 384 |
| 413844008 | chronic myocardial ischemia | STTC | `ISC_` | 0 | 0 | 0 | 161 |
| 426434006 | anterior ischemia | STTC | `ISCAN`/`ISCAS` | 281 | 0 | 0 | 0 |
| 425419005 | inferior ischaemia | STTC | `ISCIN` | 451 | 0 | 0 | 0 |
| 425623009 | lateral ischaemia | STTC | `ISCLA` | 903 | 0 | 0 | 0 |
| 428750005 | nonspecific ST-T abnormality | STTC | `NST_` | 1883 | 1158 | 0 | 1290 |
| 111975006 | prolonged QT interval | STTC | `LNGQT` | 1391 | 57 | 337 | 4 |
| 164873001 | left ventricular hypertrophy | HYP | `LVH` | 1232 | 15 | 632 | 158 |
| 89792004 | right ventricular hypertrophy | HYP | `RVH` | 86 | 4 | 106 | 20 |
| 266249003 | ventricular hypertrophy (unspec.) | HYP | `LVH`/`RVH` | 71 | 0 | 0 | 5 |
| 67741000119109 | left atrial enlargement | HYP | `LAO/LAE` | 870 | 0 | 1 | 1 |
| 446813000 | left atrial hypertrophy | HYP | `LAO/LAE` | 0 | 0 | 8 | 40 |
| 446358003 | right atrial hypertrophy | HYP | `RAO/RAE` | 0 | 3 | 33 | 18 |
| 195126007 | atrial hypertrophy (unspec.) | HYP | `LAO/LAE`/`RAO/RAE` | 60 | 0 | 0 | 2 |

Two things worth noticing in this table. `prolonged QT interval` → STTC is easy
to miss and contributes 1,391 Georgia records, because PTB-XL puts `LNGQT` in
STTC rather than treating it as an interval measurement. And the CD column is the
only one that is well populated across every source, which is the first hint at
the conclusion in §5.

### 3.3 Tier B — judgement calls that must be decided in advance

Each of these has a plausible argument on both sides. My proposal is the
right-hand column; what matters more than the choice is that it is committed to
before any inference is run, and that a sensitivity analysis reports the result
under both choices.

| SNOMED | Name | Total | The argument | Proposal |
|---|---|---:|---|---|
| 426783006 | sinus rhythm | 28,971 | The only available handle on NORM in SNOMED-coded sets. But it asserts a rhythm, not normality. | **Accept as a NORM proxy only when it is the record's sole finding** (optionally alongside a benign sinus variant — see §3.5). Report NORM separately as a proxy-based result. |
| 55827005 | left ventricular high voltage | 5,401 | Corresponds to PTB-XL `VCLVH`, which is form-only and *not* HYP. But it dominates HYP-adjacent coding in Chapman (1,295) and Ningbo (4,106); excluding it leaves Chapman with 15 HYP records. | **Exclude from HYP.** Consequence: Chapman-Shaoxing cannot support a HYP evaluation at all. Report that rather than working around it. |
| 164934002 | t wave abnormal | 11,716 | PTB-XL `TAB_` is form-only, but `NDT` ("non-diagnostic T abnormalities") is STTC and the two are hard to distinguish from a code alone. | **Exclude from STTC positives**, and exclude records where it is the only finding (see §3.5). |
| 59931005 | t wave inversion | 3,989 | Same as above; `INVT` is form-only. | Exclude, same treatment. |
| 429622005 | st depression | 3,645 | `STD_` is form-only; `NST_` is STTC. The code does not say which. | Exclude, same treatment. |
| 164931005 | st elevation | 628 | `STE_` form-only. Also raises the MI/STTC boundary: PTB-XL routes subendocardial injury (`INJAS` etc.) to **MI**, not STTC. | Exclude, same treatment. |
| 164930006 | st interval abnormal | 2,276 | Same ambiguity. | Exclude, same treatment. |
| 55930002 | s t changes | 5,009 | Closest to `NST_` of any of the ST codes; "changes" reads as a diagnostic characterisation rather than a measurement. | **Accept as STTC** — but this is the weakest Tier-A-adjacent call in the document, it contributes 4,232 Ningbo records, and it must be in the sensitivity analysis. |
| 164917005 | qwave abnormal | 2,076 | `QWAVE` is form-only; Q waves are the classical MI sign but PTB-XL declines to call them MI on their own. | **Exclude from MI**, same treatment. |
| 413444003 | acute myocardial ischemia | 2 | Ischemia → STTC, but "acute" leans toward the injury patterns PTB-XL routes to MI. | Exclude on grounds of ambiguity; n=2, so it does not matter. |
| 253352002 / 253339007 | left / right atrial abnormality | 86 | "Abnormality" is weaker than "enlargement"; `LAO/LAE` is specifically overload or enlargement. | Accept as HYP, flagged. |
| 428417006 | early repolarization | 506 | A benign normal variant. Not STTC in PTB-XL's scheme; arguably compatible with NORM. | Exclude from all classes; do **not** let it block NORM. |

### 3.4 Tier C — no defensible mapping, exclude from the label vector entirely

These are not judgement calls. They have no counterpart in PTB-XL's diagnostic
hierarchy and assigning them anywhere would be fabrication.

*Rhythm findings* (PTB-XL keeps these outside the superclass system): atrial
fibrillation, atrial flutter, all sinus variants (bradycardia, tachycardia,
arrhythmia, arrest), premature atrial/ventricular/junctional/supraventricular
complexes, bigeminy and trigeminy, pacing rhythm and pacing patterns,
supraventricular and ventricular tachycardias, atrial tachycardia, AVNRT, AVRT,
junctional rhythms and escapes, idioventricular and accelerated idioventricular
rhythm, ventricular fibrillation and flutter, AV dissociation, wandering atrial
pacemaker, brady-tachy syndrome, sinoatrial block, sinus node dysfunction.

*Form findings with no diagnostic counterpart*: left and right axis deviation
(7,631 and 1,280 — the largest single Tier C group), indeterminate cardiac axis,
low QRS voltages, poor R wave progression, clockwise/counterclockwise rotation
and vectorcardiographic loop, abnormal QRS, r wave abnormal, u wave abnormal, p
wave change, tall/prolonged P wave, high T-voltage, fusion beats, fQRS, TU
fusion, shortened PR interval, prolonged PR interval, decreased QT interval.

*Clinical rather than electrocardiographic*: heart failure, coronary heart
disease, heart valve disorder, transient ischemic attack, cardiac dysrhythmia
(unspecified), Brugada pattern, ventricular pre-excitation coded separately from
WPW, arm-lead-reversal suspicion (this one should additionally trigger record
exclusion on data-quality grounds).

### 3.5 Record inclusion rule

This is the part most likely to be got wrong quietly, so it is stated as an
algorithm. For each external record, partition its codes into A (Tier A), B+
(Tier B accepted), B− (Tier B excluded), C-rhythm and C-form.

1. If A ∪ B+ is non-empty → **include**, with those superclasses positive and the
   remainder negative. This mirrors `data.py`'s `sum(axis=1) > 0` filter.
2. Else if the record has any code in B− or C-form → **exclude**. Its superclass
   status is genuinely unknown: it cannot be NORM, because a morphological
   abnormality was coded, and it cannot be scored positive for anything, because
   the code does not license it. Silently treating these as all-negative would
   inflate every AUROC by adding easy true negatives that are not actually
   negative.
3. Else if the record's only codes are `sinus rhythm` and, optionally, benign
   sinus variants (sinus bradycardia, sinus tachycardia, sinus arrhythmia, early
   repolarization) → **include as NORM**. PTB-XL routinely labels an otherwise
   unremarkable bradycardic or tachycardic tracing `NORM`, because its rhythm
   statements are independent of `NORM`.
4. Else (only C-rhythm codes, e.g. lone AF) → **exclude**, matching PTB-XL's own
   treatment.
5. Records with no codes at all → **exclude**, and count them; a large
   uncoded fraction is itself a warning about that source.

Report the exclusion counts at every step. If step 2 removes a large fraction —
and for Ningbo, with 5,167 `t wave abnormal` and 4,232 `s t changes`, it may — the
surviving cohort is enriched for cleanly-coded records and is no longer a random
sample of the source. Say so.

### 3.6 The SPH mapping, which is much cleaner

SPH uses AHA statement codes (Liu et al., *Scientific Data*, 2022). Its primary
categories align with PTB-XL superclasses almost by construction:

| AHA category | Codes | → | Notes | Records |
|---|---|---|---|---:|
| A — Normal ECG | 1 | **NORM** | An explicit positive assertion of normality, semantically identical to PTB-XL `NORM`. This is the decisive advantage. | 13,905 |
| H — AV conduction | 80–88 | **CD** | | ~370 |
| I — Intraventricular conduction | 101–108 | **CD** | | ~2,195 |
| K — Chamber enlargement | 140–143 | **HYP** | 140 LAE 19, 142 LVH 209, 143 RVH 6 | ~234 |
| L — ST-T changes | 145–155 | **STTC** | 145 ST deviation 1,829; 146 ST deviation with T change 1,063; 147 T-wave abnormality 2,218; 148 prolonged QT 24; 153 ST-T change due to VH 88; 155 early repolarization 32 | ~5,263 |
| M — Myocardial infarction | 160–166 | **MI** | 160 anterior 52, 161 inferior 120, 165 anteroseptal 91, 166 extensive anterior 7 | ~270 |
| C, D, E, F | 21–23, 30–37, 50–54, 60 | — | rhythm; Tier C, exclude | |
| J — Axis / voltage | 120–125 | — | Tier C form; exclude | |

Two residual judgement calls remain even here. AHA code 147 "T-wave abnormality"
is the same `TAB_`-versus-`NDT` ambiguity as before, but SPH places it inside the
ST-T category rather than a separate morphology category, which is a weak
argument for accepting it; I would accept it and flag it, and sensitivity-test.
And code 153, "ST-T change due to ventricular hypertrophy", is genuinely both
STTC and HYP in PTB-XL terms; at 88 records it will not move the result, and I
would map it to both.

**But note the counts.** MI ≈ 270 and HYP ≈ 234 out of 25,770. Those are small
enough that bootstrap intervals will be wide — the PTB-XL fold-10 HYP interval
already spans 0.813–0.860 at n = 262 positives, and SPH will be no better. SPH
buys correct semantics, not statistical power for the rare classes.

### 3.7 Process safeguard

Write the mapping as a committed CSV in the repository — one row per source code,
with columns for the target superclass, the tier, and a one-line justification —
**and commit it before the first external inference run**. The git timestamp then
demonstrates that the mapping was not adjusted after seeing results. Any
subsequent change to it should be a separate commit with the reason stated. This
costs nothing and is the only mechanism that makes the pre-registration claim
checkable by a reader.

Separately, before running anything, pull ~20 records per ambiguous code and plot
them. A human eyeball on 200 tracings will catch a wrong mapping that no amount
of code review will.

---

## 4. Preprocessing mismatches

Each of these can degrade measured performance without any genuine failure to
generalise, and each would look like poor generalisation in the results table.

### 4.1 Sampling rate: 500 Hz → 100 Hz

The model consumes 100 Hz. Every candidate except INCART (257 Hz) and PTB
Diagnostic (1000 Hz) is 500 Hz.

The failure mode is naive decimation. Taking every fifth sample without an
anti-alias filter folds everything above 50 Hz back into the passband — pacing
spikes, mains interference at 50/60 Hz, EMG. Mains interference is the nastiest
case, because 50 Hz aliases to exactly 50 Hz and 60 Hz aliases to 40 Hz, landing
inside the diagnostic band, and because different countries and different
recording equipment give different amounts of it. That would produce a
source-dependent artefact indistinguishable from a real cohort effect.

Use `scipy.signal.decimate(x, 5, ftype='fir', zero_phase=True)` for 500 Hz, and
`scipy.signal.resample_poly(x, 100, 257)` for INCART if it is ever used. Prefer
FIR with `zero_phase=True` over the default IIR: an IIR filter introduces
phase distortion that shifts ST segments and T waves relative to the QRS, which
is precisely the morphology STTC and MI depend on.

**Honest caveat.** PTB-XL's own 100 Hz records were downsampled by the dataset
authors, and I do not know what filter they used. So even a correctly implemented
decimation leaves a residual mismatch between the external 100 Hz signals and the
100 Hz signals the model trained on. This is a real, unquantified contribution to
whatever drop is observed. It can be bounded with a cheap control: PTB-XL also
ships 500 Hz records. Decimate a sample of PTB-XL fold 10 from 500 Hz using your
pipeline, run the model, and compare against the published 100 Hz result. Any gap
is your resampling artefact, measured on data with no domain shift at all. This
control costs one inference pass over 2,158 records and should be run **before**
touching any external data. Note that the repository's `data.py` currently
extracts only `records100/` from the archive, so this requires re-extracting the
500 Hz members — plan for the extra disk.

### 4.2 Recording length

PTB-XL is uniformly 10 s = 1,000 samples. The external sets are not: Georgia is
5–10 s, SPH is 10–60 s, CPSC-Extra runs to 144 s.

Architecturally this is not a problem. Both `ECGEncoder` and `XResNet1d` end in
`nn.AdaptiveAvgPool1d(1)` before the embedding projection, so any input length is
accepted. That is a trap rather than a convenience: the model will happily
consume a 60-second record and return a number, but its entire training
distribution is 10 seconds, and global average pooling over 60 seconds dilutes a
transient finding by a factor of six.

Handle it explicitly:

- **Longer than 10 s** — tile into non-overlapping 10 s windows, run each, and
  aggregate per class across windows. Choose the aggregation rule in advance.
  **Maximum** is the right default for findings that may be intermittent, and
  matches how a reader scans a long strip; mean is defensible for NORM. Whatever
  is chosen, apply it uniformly and report it. Do not evaluate both and keep the
  better one.
- **Shorter than 10 s** — Georgia contains records down to 5 s. Padding invents
  signal. Reflection padding invents plausible-looking signal, which is worse
  because it is harder to notice. Prefer **excluding** records under 10 s and
  reporting the count. If too many are lost, the fallback is to run the model on
  the short record as-is (the adaptive pooling permits it) and report those
  records as a separate stratum. **I do not know what fraction of Georgia is
  under 10 s** — check it before deciding.

### 4.3 Lead ordering

`data.py` defines `LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1"…"V6"]`
and `load_ecg` reads channels in file order via `wfdb.rdsamp`, implicitly trusting
that PTB-XL's file order matches. For PTB-XL it does. For an external source it
must be verified rather than assumed.

Read the signal names from the WFDB header (or the SPH metadata) and permute
explicitly by name into the PTB-XL order. Assert that all twelve expected names
are present and that there are exactly twelve, and fail loudly on any record that
does not match rather than skipping it silently.

This matters more than it sounds. A silent lead permutation is the classic cause
of a catastrophic external result — it would hit MI and STTC hardest, because
those classes are inherently regional (anterior versus inferior versus lateral),
while CD would survive better because bundle branch morphology is visible across
many leads. **If the eventual results show MI and STTC collapsing while CD holds
up, suspect lead ordering before concluding anything about generalisation.**

Also normalise naming variants: `AVR` versus `aVR`, `MDC_ECG_LEAD_I`-style
identifiers in some exports.

### 4.4 Amplitude and units

This one is mostly already handled, and the reason is worth knowing.

`load_ecg` z-scores each lead independently per record:

```python
mean = signal.mean(axis=1, keepdims=True)
std  = signal.std(axis=1, keepdims=True) + 1e-6
return (signal - mean) / std
```

Combined with `wfdb.rdsamp`, which applies the header's gain and baseline to
return physical units, this makes the pipeline invariant to ADC scaling, to
whether a source stores microvolts or millivolts, and to per-lead gain
differences. Amplitude mismatch is therefore the *least* dangerous of the four
preprocessing issues for WFDB sources. Do not "fix" it by adding a global scaling
step.

Three real risks remain.

**Flat or near-flat leads.** If a lead is disconnected, `std ≈ 0`, and the `+1e-6`
floor amplifies pure quantisation noise into a full-scale garbage channel. In
PTB-XL this is presumably rare; in a hospital export it may not be. Add an
explicit check — count records where any lead has `std` below a threshold in
physical units, exclude them, and report the number as a data-quality figure.

**SPH is HDF5, not WFDB.** It stores a 12 × L array at 16-bit precision, and
there is no `rdsamp` to apply a gain for you. **I did not verify what physical
unit or resolution SPH's stored integers correspond to.** Before trusting
anything, load a handful of records, apply whatever gain the SPH documentation
specifies, and check that R-wave amplitudes land in a physiologically sane range
(roughly 0.5–2 mV in lead II for a normal tracing). If they do not, the gain is
wrong. Because of the per-lead z-scoring this error would not produce an obvious
crash — it would produce a slightly degraded result and no warning at all.

**Per-lead z-scoring deletes absolute voltage, and HYP depends on absolute
voltage.** Left ventricular hypertrophy is diagnosed by voltage criteria — Sokolow-
Lyon, Cornell — which are amplitude thresholds in millivolts. The current
preprocessing removes exactly that information before the model sees it. This is
a pre-existing limitation, not an external-validation issue, but it matters here
for interpretation: **if external HYP performance is poor, the cause is confounded
between domain shift, the `LVHV`-versus-`LVH` mapping decision in §3.3, and a
preprocessing choice that discards the diagnostic feature.** Do not attribute an
external HYP drop to generalisation failure. This also suggests a separate,
genuinely useful experiment: retrain with per-record rather than per-lead
normalisation, which preserves inter-lead amplitude ratios, and see whether HYP
improves in-distribution.

### 4.5 Filtering and front-end differences

PTB-XL was recorded on Schiller equipment with its own front-end filter settings.
Other sources will differ in high-pass corner (baseline wander) and in notch
filtering. There is no principled correction, because the training distribution
already contains whatever PTB-XL's front end did.

The right move is to add no extra filtering — anything applied to the external
data and not to the training data is itself a domain shift — but to *measure* the
difference: compare the mean power spectral density of the external cohort
against PTB-XL fold 10, per lead, and include the plot. If the low-frequency
content differs markedly, that is a documented, visible confound rather than an
invisible one. This is cheap and it converts an unknown into a stated limitation.

---

## 5. What a fair result looks like, stated in advance

### 5.1 Published anchors

The most directly relevant published result is **Leinonen, Wong, Vasankari,
Wahab, Nadarajah, Kaisti and Airola, "Empirical investigation of multi-source
cross-validation in clinical ECG classification", *Computers in Biology and
Medicine* 183:109271, 2024**. It is relevant because it uses this exact family of
datasets — Chapman-Shaoxing/Ningbo, CPSC/CPSC-Extra, G12EC, PTB/PTB-XL and
Shandong Provincial Hospital, 103,438 recordings across five sources — and
measures the gap between within-source k-fold estimates and leave-source-out
estimates over a 17-label set.

Its reported figures: 4-fold cross-validation overestimates macro-AUC by **+0.040
to +0.112** in the multi-source setting, while leave-source-out is close to
unbiased (−0.006 to +0.004). In single-source experiments the k-fold optimism
averaged **0.04 to 0.08 AUC points**.

That is the number to plan against. The Cardiac Nexus model is evaluated on
PTB-XL's official patient-wise fold 10, which is a within-source held-out split —
better than random k-fold, because it prevents patient leakage, but still
same-source. It is subject to exactly the optimism Leinonen et al. quantify.

Second anchor, already cited in the README: **Strodthoff, Wagner, Schaeffter and
Samek, "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL",
*IEEE Journal of Biomedical and Health Informatics* 25(5), 2021**, reporting 0.928
macro AUROC on the PTB-XL superdiagnostic task with `xresnet1d101`. That paper
also benchmarks ICBEB2018 (CPSC) and discusses transfer from PTB-XL-pretrained
classifiers, but its ICBEB evaluation is a fine-tuned model on ICBEB's own nine
classes, not a zero-shot superclass transfer — **it is not a measurement of the
drop being predicted here**, and should not be cited as if it were.

I looked for a published zero-shot PTB-XL-superclass-to-external-cohort AUROC and
**did not find one I could verify**. Several search results asserted specific
figures, but I could not confirm them against the papers themselves, so they are
not reproduced here. If the Cardiac Nexus result is the first clean measurement of
this specific quantity, that is worth saying — and it raises the bar for
methodological care rather than lowering it.

Worth naming for what it is not: **Ribeiro et al., "Automatic diagnosis of the
12-lead ECG using a deep neural network", *Nature Communications* 11:1760, 2020**
is the standard reference for large-scale ECG deep learning, but its evaluation
set is drawn from the same telehealth network as its training data and
re-annotated by cardiologists. It measures annotation quality, not cross-source
generalisation. Cite it for the former only.

### 5.2 Pre-registered interpretation bands

Committed before the first external inference run. Current in-distribution macro
AUROC is 0.9107.

| External macro AUROC | Interpretation |
|---|---|
| **0.83 – 0.88** | **The expected outcome.** A drop of 0.03–0.08 sits squarely inside the optimism range Leinonen et al. measured on these datasets. Report it as a successful external validation and as evidence that the fold-10 number was somewhat optimistic — which is the honest, useful finding. |
| 0.88 – 0.91 | Better than published cross-source expectations. Do not celebrate; audit. Check first for PTB-XL contamination in the cohort (§1), then for a mapping that has quietly become easy — e.g. a cohort dominated by CD after the §3.5 exclusions, when CD is the least ambiguous class. A "good" result here is more likely a bug than a triumph. |
| 0.78 – 0.83 | Larger than expected but not implausible, particularly for a cross-continental shift (PTB-XL is German; Chapman, Ningbo and SPH are Chinese). Report it as-is, with the §4 controls to show it is not preprocessing. Do not tune anything to improve it. |
| **below 0.75** | **Treat as a bug until proven otherwise.** Domain shift of this size would be extraordinary for a task with this much shared morphology. In order: lead ordering (§4.3), then resampling (§4.1), then label mapping polarity, then the record inclusion rule. Run the PTB-XL 500 Hz self-control from §4.1 before writing a single word of interpretation. |

Per class, again in advance:

- **CD is expected to hold up best.** Bundle branch blocks and AV blocks are the
  most objectively defined findings, coded consistently across every source, and
  Tier A covers them with no judgement calls. If CD drops sharply, something is
  wrong with the pipeline, not the cohort.
- **NORM is expected to be the most mapping-sensitive**, because outside SPH it is
  a proxy for "nothing was coded". If a source under-codes, genuinely abnormal
  ECGs enter the NORM-positive group and NORM AUROC falls for reasons that have
  nothing to do with the model.
- **HYP is expected to be worst and least interpretable**, for the three
  confounded reasons in §4.4. Report it, but do not draw conclusions from it.
- **MI will have intervals too wide to be useful** on Georgia, Chapman or SPH.
  State the positive count next to the AUROC every time so no reader mistakes a
  40-positive estimate for a measurement.
- **Any class falling below about 0.60 is a bug.** Real domain shift degrades a
  genuine signal; it does not reduce it to chance. A class near or below 0.50
  means an inverted or misassigned label.

### 5.3 Reporting requirements

- **Prevalence table, external cohort beside PTB-XL fold 10, per class.** AUROC is
  prevalence-insensitive; average precision is not. A large AP drop with a small
  AUROC drop is usually prevalence, and saying so pre-empts an obvious objection.
- **Bootstrap intervals on the external set**, using the existing
  `bootstrap_metric`. With MI positives in the tens, the interval is the result.
- **Calibration reported twice.** The vector scaler in `evaluate.py` was fitted on
  PTB-XL fold 9 and will not transfer, because base rates differ. Report ECE with
  the PTB-XL-fitted scaler applied unchanged — that is the honest number, and it
  will be bad — and, separately, ECE after refitting on a held-out slice of the
  external cohort, as an upper bound on what recalibration could recover. Never
  report only the refitted figure.
- **A full accounting of excluded records**, by exclusion reason, with the
  surviving fraction stated. If step 2 of §3.5 removes 40% of a source, the
  remaining cohort is not that source and the write-up must say so.
- **The §4.1 PTB-XL self-control**, reported as a number, so a reader can subtract
  the resampling artefact from the observed drop.
- **No threshold tuning, no test-time augmentation, no per-source calibration
  chosen after the fact.** One model, one mapping, one pass.

---

## 6. Recommended sequence

**Phase 0 — controls, no external data.** Re-extract PTB-XL 500 Hz for fold 10,
decimate with the intended pipeline, re-run, and compare against 0.9107. This
measures the resampling artefact on data with zero domain shift, and it exercises
the entire new code path before any confounds exist. Cheap, and it is the single
highest-value step in the plan.

**Phase 1 — SPH.** Download from Figshare (DOI 10.6084/m9.figshare.c.5779802.v1),
write the HDF5 loader, verify the gain empirically (§4.4), map AHA codes per §3.6,
commit the mapping, run.

**Phase 2 — Georgia.** From the Challenge 2020/2021 bundle (`WFDB_Ga` /
`PhysioNetChallenge2020_Training_E`). WFDB, so the existing `wfdb` dependency is
enough. This adds a US cohort and tests whether the drop is China-specific.
Georgia covers NORM (proxied), STTC, CD and HYP; it does **not** cover MI, and
that should be stated as a blank in the results table rather than filled with a
7-positive estimate.

**Phase 3, optional — Chapman-Shaoxing / Ningbo.** Volume, from `ecg-arrhythmia`
v1.0.0. Useful for tightening CD and STTC intervals. HYP is unavailable after
excluding `left ventricular high voltage`, and MI is unavailable at 40 and 189
positives.

**Phase 4, optional — CPSC-Extra for MI only.** The only public source with a
substantial MI population under a mappable code set. A single-class,
single-source result, reported as such.

**Effort, honestly.** The loader and resampling are perhaps one to two days. The
evaluation harness largely exists. The label mapping is the expensive part: two to
four days including the manual review of sampled tracings for each ambiguous code,
and it should not be compressed, because it is the part that determines whether
the result means anything. The inference itself is minutes.

**Machine constraints.** All of this is I/O and light compute apart from the
inference pass. Chapman + Ningbo decoded to a 100 Hz float32 cache would be
45,152 × 12 × 1,000 × 4 bytes ≈ 2.2 GB (arithmetic estimate), which on an 8 GB M1
must be memory-mapped, as `build_signal_cache` already does, and must not run
concurrently with training. SPH at 25,770 records of 10–60 s is larger before
windowing; decode straight to windows rather than caching full-length records.

**A note on caching.** `build_signal_cache` writes to a fixed
`signals_100hz.npy` keyed on the PTB-XL record index. The external path needs its
own cache files and its own index, or it will silently invalidate and rebuild the
PTB-XL cache on every alternation.

---

## 7. What could go wrong

**The cohort shrinks to the point of meaninglessness.** The §3.5 exclusion rule is
strict by design. If Ningbo loses most of its records to `t wave abnormal` and `s
t changes`, what survives is a biased sub-cohort. This is a real possibility and
the mitigation is to report the survival fraction prominently, not to loosen the
rule after seeing it.

**The mapping is quietly wrong in one direction.** The Tier B decisions all lean
toward exclusion, which means external positives are under-counted, which biases
AUROC in an unpredictable direction — under-counted positives become false
negatives in the label vector and depress apparent performance. The sensitivity
analysis under the opposite Tier B choices is not optional; it is the only thing
that bounds this.

**The result is good and nobody believes it anyway.** If the external macro AUROC
lands at 0.88, a sceptical reader's first thought will be contamination. Pre-empt
it: state the excluded sources explicitly, state that PTB-XL and PTB were removed
from any challenge-derived cohort, and publish the record-id manifest.

**The result is bad and the cause is unattributable.** The most likely bad outcome
is not a low number but a low number with four plausible explanations. That is
what §4's controls exist for. Run them first, so that when a drop appears there is
already a measured bound on how much of it is resampling.

**The honest possibility that the whole thing is negative.** It may turn out that
no public dataset supports a five-superclass external validation, and that the
defensible deliverable is a three-class result (NORM, STTC, CD) on SPH and
Georgia, with MI and HYP declared not externally evaluable on open data. That
would be a genuinely useful contribution and a much stronger statement than a
five-class table built on a mapping that does not hold up. It should be treated as
an acceptable outcome from the start, not as a failure discovered late.

---

## References

- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., Schaeffter, T. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7:154, 2020.
- Strodthoff, N., Wagner, P., Schaeffter, T., Samek, W. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE Journal of Biomedical and Health Informatics* 25(5):1519–1528, 2021. (arXiv:2004.13701)
- Leinonen, T., Wong, D., Vasankari, A., Wahab, A., Nadarajah, R., Kaisti, M., Airola, A. "Empirical investigation of multi-source cross-validation in clinical ECG classification." *Computers in Biology and Medicine* 183:109271, 2024. (arXiv:2403.15012)
- Zheng, J., Zhang, J., Danioko, S., Yao, H., Guo, H., Rakovski, C. "A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients." *Scientific Data* 7:48, 2020.
- Liu, H., Chen, D., Chen, D., Zhang, X., Li, H., Bian, L., Shu, M., Wang, Y. "A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements." *Scientific Data* 9:272, 2022.
- Ribeiro, A. H., Ribeiro, M. H., Paixão, G. M. M., et al. "Automatic diagnosis of the 12-lead ECG using a deep neural network." *Nature Communications* 11:1760, 2020. (Cited only as a same-source annotation-quality evaluation, not as cross-source generalisation evidence.)
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. "On Calibration of Modern Neural Networks." *ICML*, 2017. (Already cited in `evaluate.py` for temperature scaling.)
- PhysioNet/CinC Challenge 2020 and 2021 project pages, and the `dx_mapping_scored.csv` / `dx_mapping_unscored.csv` tables in the `physionetchallenges/evaluation-2021` repository, from which all per-code, per-database counts in §3 are taken.
- PTB-XL `scp_statements.csv` v1.0.3, read locally, from which the superclass membership in §3.1 is taken.
