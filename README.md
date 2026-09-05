# Nexus Research Documentation

Research, experiment, and project documentation for the Cardiac Nexus initiative.

## Purpose

This repository maintains the written record of the project’s research direction, methodology, datasets, experiments, results, limitations, and development decisions. It is intended to keep the work organized, reproducible, and easy to review as the project develops.

## Current contents

```text
.
├── experiments/
│   ├── 001_ecg_mi_baseline.md
│   └── 002_ecg_explainability_integrated_gradients.md
└── README.md
```

### Experiment records

Experiment records include:

- Objective and research question
- Dataset and label definitions
- Data-splitting strategy
- Model and training configuration
- Evaluation metrics
- Results and visualizations
- Limitations
- Reproduction instructions
- Follow-up questions and next experiments

## Completed experiments

| Record | Subject | Outcome |
| --- | --- | --- |
| [001](experiments/001_ecg_mi_baseline.md) | 12-lead ECG baseline, MI versus non-MI, PTB-XL | Test AUROC 0.920 |
| [002](experiments/002_ecg_explainability_integrated_gradients.md) | Integrated Gradients attribution over the baseline | Attribution computed and rendered for four test recordings |

Together these cover the two methods the project is built on: a deep learning classifier, and an explanation of what that classifier responded to.

## Documentation standards

Each experiment should identify:

1. The date and objective.
2. The dataset version and source.
3. The target definition.
4. The train, validation, and test split.
5. The model and configuration used.
6. The evaluation metrics and test results.
7. Limitations and unresolved concerns.
8. The code and artifacts required for reproduction.

Results should be recorded before changing the experiment configuration. This makes comparisons between experiments clear and traceable.

## Data and privacy

Patient-identifiable information, confidential hospital records, credentials, and private clinical files must not be committed to this repository. Public datasets should be referenced by their official source and handled according to their applicable terms.

## Project repositories

The project is divided across three repositories. Each owns one kind of work, and work should be committed to the repository that owns it rather than duplicated.

| Repository | Owns | Contains |
| --- | --- | --- |
| [Nexus AI Engine](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine) | Code and artifacts | Notebooks, training and evaluation code, preprocessing, model checkpoints, raw metric records, generated figures |
| [Nexus Research Docs](https://github.com/Cardiac-Nexus-Lab/nexus-research-docs) | The written record | Experiment reports, methodology, decisions, literature notes, ethics and regulatory planning, roadmap |
| [Nexus Web Portal](https://github.com/Cardiac-Nexus-Lab/nexus-web-portal) | The application | User-facing interface, inference integration, deployment configuration |

The boundary between the first two repositories is the distinction between an artifact and its interpretation. A metric printed by a notebook, the checkpoint that produced it, and the figure it rendered are artifacts and belong in the AI engine. The report that explains what the experiment was, why it was configured that way, what the result means, and where it falls short is interpretation and belongs here.

The web portal consumes artifacts from the AI engine rather than storing its own copies of them.

## Project status

Two experiments are recorded: an ECG baseline classifier and an explainability evaluation over it. Neither has undergone external validation, calibration analysis, or clinical review.

Planned next records cover multi-label diagnostic classification, additional modalities, local data planning, and deployment decisions.
