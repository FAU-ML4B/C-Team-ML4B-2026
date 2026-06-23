# SafeStep

A prototype fall-detection system using smartphone IMU data (accelerometer + gyroscope + gravity), built as the ML4B project (SoSe 2026, FAU Erlangen-Nürnberg).

**Live demo:** [c-team-ml4b-fall-detection.streamlit.app](https://c-team-ml4b-fall-detection.streamlit.app)

Upload a Sensor Logger ZIP file in the web app and receive a fall prediction in seconds.

## What the system does

The pipeline takes a Sensor Logger recording, extracts seventy hand-crafted features from the IMU signal, and runs a two-stage Random Forest classifier:

- **Stage 1** — binary decision: did a fall occur, or is this an activity of daily living (ADL)?
- **Stage 2** — if Stage 1 returns fall, predicts the fall direction (backwards / forward / left / right / stumble).

Both stages were evaluated using Leave-One-Subject-Out cross-validation across the three team members.

## Main results

| Metric                  | Score | Random baseline |
|-------------------------|-------|-----------------|
| Stage 1 LOSO macro-F1   | 0.987 | 0.500           |
| Stage 2 LOSO macro-F1   | 0.258 | 0.200           |

Stage 1 generalises reliably to unseen subjects across three phone placements (pocket, hand, chest), with zero false positives on the ADL class. Stage 2 sits only marginally above the random baseline; we discuss the data-side reasons for this in `project.md` §3.3 and treat the fall-type output as auxiliary information rather than a reliable diagnostic.

## Quick start

The system can be used in three modes: through the deployed Streamlit app, locally for inference, or as a full training pipeline.

### 1. Use the deployed app
Open the [live demo](https://c-team-ml4b-fall-detection.streamlit.app), upload a Sensor Logger ZIP, see the prediction. No setup required.

### 2. Run locally
```bash
git clone https://github.com/FAU-ML4B/C-Team-ML4B-2026.git
cd C-Team-ML4B-2026
conda env create -f docs/environment.yml
conda activate ml4b-falldetection
streamlit run app/streamlit_app.py
```

### 3. Retrain the models on new data
Place new Sensor Logger recordings into `data/raw/`, following the naming schema `{subject}_{placement}_{class}_{trial}_{date}`, then:

```bash
python -m src.build_dataset data/raw data/processed
python -m src.train
```

The trained models are written to `models/`, confusion matrices to `data/processed/confusion_stage{1,2}.png`, and a LOSO report to `data/processed/loso_report.txt`. The Streamlit app picks up the new models automatically on next restart.

## Repository structure

```text
C-Team-ML4B-2026/
├── app/
│   └── streamlit_app.py        Streamlit front-end
├── src/
│   ├── parsing.py              Sensor Logger CSV reader + resampling
│   ├── preprocessing.py        Butterworth filter, windowing, label parsing
│   ├── features.py             Seventy hand-crafted IMU features
│   ├── build_dataset.py        Builds Stage 1 + Stage 2 datasets from data/raw/
│   ├── train.py                Trains and evaluates both stages with LOSO
│   └── inference.py            predict(folder_path) used by the Streamlit app
├── models/
│   ├── stage1_binary.joblib    Trained Stage 1 model
│   └── stage2_falltype.joblib  Trained Stage 2 model
├── docs/
│   ├── environment.yml         Conda environment specification
│   ├── figures/                Confusion matrices for the report
│   └── loso_report.txt         Full per-fold classification reports
├── tests/
│   └── test_parsing.py         Smoke tests for the parsing pipeline
├── project.md                  Full project report
└── README.md                   This file
```

## Dataset

The dataset consists of 290 self-collected Sensor Logger recordings produced by the three team members between 23 May and 04 June 2026, distributed across three smartphone placements (trouser pocket, hand-held, chest pocket) and twelve activity classes (five fall types and seven ADL types). Full collection protocol and class distribution are documented in `project.md` §3.2.

Raw and processed data are excluded from version control via `.gitignore`. To reproduce the dataset, request the raw recordings from the team.

## Limitations

The system was validated on three young healthy adults, on iPhone devices only, with all falls onto a foam mattress. Behaviour on elderly subjects, on Android devices with different axis conventions, or on hard-floor impacts is outside the training distribution and unknown. See `project.md` §5 for a detailed discussion and proposed future work.

## Team

C-Team, ML4B SoSe 2026, FAU Erlangen-Nürnberg:

- Felix Geier — data pipeline, modelling (`src/`)
- Lea Hermann — data collection, project documentation
- Leopold Eberlein — Streamlit application, project documentation

Tutor: Markus Walk. Final submission: 12 June 2026.
