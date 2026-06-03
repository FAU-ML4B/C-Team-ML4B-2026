"""
Honest model comparison for Stage 2 (5-class fall type classification).

Tries four classifiers with sensible defaults and reports LOSO macro-F1
for each. The intent is to find a better baseline than Random Forest,
NOT to extensively tune any one of them (we don't have enough data
for that to be meaningful).

Run:
    python -m src.tune_stage2
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import extract_features_batch

PROCESSED_DIR = Path("data/processed")
RANDOM_STATE = 42


def loso_macro_f1(clf, X, y, groups):
    """Run LOSO and return (mean, per-fold list)."""
    logo = LeaveOneGroupOut()
    f1s = []
    for tr, te in logo.split(X, y, groups):
        clf_fold = clf  # the pipeline is stateless once constructed; refit each fold
        clf_fold.fit(X[tr], y[tr])
        yp = clf_fold.predict(X[te])
        f1s.append(f1_score(y[te], yp, average="macro", zero_division=0))
    return float(np.mean(f1s)), f1s


def main():
    X_raw = np.load(PROCESSED_DIR / "stage2" / "X.npy")
    meta = pd.read_parquet(PROCESSED_DIR / "stage2" / "meta.parquet")
    X, names = extract_features_batch(X_raw)

    class_names = sorted(meta["class_name"].unique())
    y = meta["class_name"].map({n: i for i, n in enumerate(class_names)}).values
    groups = meta["subject"].values

    print(f"Stage-2 dataset: {X.shape[0]} samples, {X.shape[1]} features, classes = {class_names}")
    print(f"Subjects: {sorted(set(groups))}")
    print(f"Per-class counts: {meta['class_name'].value_counts().to_dict()}\n")

    candidates = [
        (
            "RandomForest (baseline)",
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
        ),
        (
            "RandomForest (shallow, max_depth=5)",
            RandomForestClassifier(
                n_estimators=300, max_depth=5, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "SVM RBF (scaled + top-20 features)",
            Pipeline([
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif, k=20)),
                ("svc", SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)),
            ]),
        ),
        (
            "LogReg (scaled + top-15 features)",
            Pipeline([
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif, k=15)),
                ("lr", LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000,
                    random_state=RANDOM_STATE,
                )),
            ]),
        ),
    ]

    print(f"{'model':<45s}  mean    per-fold")
    print("-" * 80)
    results = []
    for name, clf in candidates:
        mean, folds = loso_macro_f1(clf, X, y, groups)
        results.append((name, mean, folds))
        folds_str = ", ".join(f"{f:.3f}" for f in folds)
        print(f"{name:<45s}  {mean:.3f}   [{folds_str}]")

    print("\n" + "=" * 80)
    best = max(results, key=lambda r: r[1])
    print(f"Best: {best[0]}  (mean macro-F1 = {best[1]:.3f})")
    print()
    print("Interpretation guide:")
    print(f"  Random baseline (5 classes) = 0.20")
    print(f"  Anything < 0.25            = no signal, model is essentially guessing")
    print(f"  0.30 - 0.50                = real signal, but limited generalisation across subjects")
    print(f"  > 0.50                     = solid")


if __name__ == "__main__":
    main()
