# Nexus Research Documentation

Research, experiment, and project documentation for the Cardiac Nexus initiative.

## Purpose

This repository maintains the written record of the project’s research direction, methodology, datasets, experiments, results, limitations, and development decisions. It is intended to keep the work organized, reproducible, and easy to review as the project develops.

## Current contents

```text
.
├── experiments/
│   └── 001_ecg_mi_baseline.md
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

## Completed baseline

The first documented experiment evaluates a 12-lead ECG baseline for MI versus non-MI classification using the PTB-XL dataset.

Read the report: [Experiment 001 — ECG MI versus Non-MI Baseline](experiments/001_ecg_mi_baseline.md)

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

- [Nexus AI Engine](https://github.com/Cardiac-Nexus-Lab/nexus-ai-engine) — signal-processing and model-development code
- [Nexus Web Portal](https://github.com/Cardiac-Nexus-Lab/nexus-web-portal) — user-facing application and later inference integration

## Project status

The documentation repository is currently recording the initial ECG baseline and its evaluation. Future records will cover model comparisons, interpretation methods, local data planning, validation, and deployment decisions.
