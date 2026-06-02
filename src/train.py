"""
Train and evaluate the two-stage fall detection model.

Stage 1: binary fall vs ADL              -- primary result of the project
Stage 2: 5-class fall type classification -- reported with honest limitations

Both stages use Random Forest with class_weight="balanced". LOSO
cross-validation is used to report subject-generalising performance.

Outputs:
  - models/stage1_binary.joblib
  - models/stage2_falltype.joblib
  - data/processed/loso_report.txt
  - data/processed/confusion_stage1.png, confusion_stage2.png
"""
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut

from src.features import extract_features_batch

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
N_ESTIMATORS = 300
RANDOM_STATE = 42


def _load(name: str) -> Tuple[np.ndarray, pd.DataFrame]:
    base = PROCESSED_DIR / name
    return np.load(base / "X.npy"), pd.read_parquet(base / "meta.parquet")


def _loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_names,
    title: str,
) -> "tuple[float, list[float], list[str]]":
    logo = LeaveOneGroupOut()
    reports = []
    fold_f1s = []
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(logo.split(X, y, groups), start=1):
        held = groups[te[0]]
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])
        f1 = f1_score(y[te], yp, average="macro", zero_division=0)
        fold_f1s.append(f1)
        rep = classification_report(
            y[te], yp,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
        )
        reports.append(f"\n--- Fold {fold}: held out = '{held}'  (macro-F1 = {f1:.3f}) ---\n{rep}")
        all_true.append(y[te])
        all_pred.append(yp)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true_all, y_pred_all,
        display_labels=label_names, normalize="true",
        cmap="Blues", xticks_rotation=45,
    )
    disp.figure_.suptitle(f"{title}\nMean LOSO macro-F1 = {np.mean(fold_f1s):.3f}")
    disp.figure_.tight_layout()
    return float(np.mean(fold_f1s)), fold_f1s, reports


def train_stage1():
    print("\n=== Stage 1: binary fall vs ADL ===")
    X_raw, meta = _load("stage1")
    print(f"  samples: {len(X_raw)}  subjects: {sorted(meta['subject'].unique())}")
    print(f"  positive (fall) ratio: {meta['binary_label'].mean():.3f}")

    X, feat_names = extract_features_batch(X_raw)
    y = meta["binary_label"].values.astype(int)
    groups = meta["subject"].values
    print(f"  features per sample: {X.shape[1]}")

    mean_f1, fold_f1s, reports = _loso(
        X, y, groups,
        label_names=["adl", "fall"],
        title="Stage 1 - Binary fall vs ADL",
    )
    plt.savefig(PROCESSED_DIR / "confusion_stage1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  mean LOSO macro-F1: {mean_f1:.3f}  (folds: {[f'{f:.3f}' for f in fold_f1s]})")

    final = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    final.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final, "feature_names": feat_names},
        MODELS_DIR / "stage1_binary.joblib",
    )
    return reports, mean_f1


def train_stage2():
    print("\n=== Stage 2: 5-class fall type ===")
    X_raw, meta = _load("stage2")
    print(f"  samples: {len(X_raw)}  subjects: {sorted(meta['subject'].unique())}")
    print(f"  class distribution:\n{meta['class_name'].value_counts().to_string()}")

    X, feat_names = extract_features_batch(X_raw)
    class_names = sorted(meta["class_name"].unique().tolist())
    y = meta["class_name"].map({n: i for i, n in enumerate(class_names)}).values
    groups = meta["subject"].values

    mean_f1, fold_f1s, reports = _loso(
        X, y, groups,
        label_names=class_names,
        title="Stage 2 - Fall-type classification",
    )
    plt.savefig(PROCESSED_DIR / "confusion_stage2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  mean LOSO macro-F1: {mean_f1:.3f}  (folds: {[f'{f:.3f}' for f in fold_f1s]})")
    print(f"  random baseline   : {1.0 / len(class_names):.3f}  (5 classes)")

    final = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    final.fit(X, y)
    joblib.dump(
        {"model": final, "feature_names": feat_names, "class_names": class_names},
        MODELS_DIR / "stage2_falltype.joblib",
    )
    return reports, mean_f1


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reports_1, f1_1 = train_stage1()
    reports_2, f1_2 = train_stage2()

    report_path = PROCESSED_DIR / "loso_report.txt"
    with report_path.open("w") as f:
        f.write("LOSO Cross-Validation Report\n" + "=" * 60 + "\n")
        f.write(f"\nStage 1 (binary fall vs ADL):   mean macro-F1 = {f1_1:.3f}\n")
        f.write(f"   (random baseline = 0.500)\n")
        f.write("\n".join(reports_1))
        f.write(f"\n\nStage 2 (5-class fall type):    mean macro-F1 = {f1_2:.3f}\n")
        f.write(f"   (random baseline = 0.200)\n")
        f.write("\n".join(reports_2))

    print(f"\nReport: {report_path}")
    print(f"Plots:  {PROCESSED_DIR}/confusion_stage{{1,2}}.png")
    print(f"Models: {MODELS_DIR}/")


if __name__ == "__main__":
    main()
