"""
Honest comparison: does merging fall classes into 3 categories produce a
classifier that actually beats random guessing?

5-class setup: backwards / forward / left / right / stumble    (random baseline = 0.20)
3-class setup: frontal (back+forward) / sideways (left+right) / stumble  (random baseline = 0.33)

The same five classifier families are evaluated on both setups using LOSO.

Run:
    python -m src.tune_stage2_3class
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

# 5 -> 3 class merge
MERGE_MAP = {
    "forward": "frontal",
    "backwards": "frontal",
    "left": "sideways",
    "right": "sideways",
    "stumble": "stumble",
}


def loso_macro_f1(clf_factory, X, y, groups):
    """clf_factory is a callable returning a fresh classifier each fold."""
    logo = LeaveOneGroupOut()
    f1s = []
    for tr, te in logo.split(X, y, groups):
        clf = clf_factory()
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])
        f1s.append(f1_score(y[te], yp, average="macro", zero_division=0))
    return float(np.mean(f1s)), f1s


def make_candidates():
    """Factories so each fold gets a fresh, unfit classifier."""
    return [
        ("RandomForest", lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
        ("RF shallow (depth=5)", lambda: RandomForestClassifier(
            n_estimators=300, max_depth=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
        ("GradientBoosting", lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE,
        )),
        ("SVM RBF (top-20)", lambda: Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=20)),
            ("svc", SVC(C=1.0, gamma="scale", class_weight="balanced",
                        random_state=RANDOM_STATE)),
        ])),
        ("LogReg (top-15)", lambda: Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=15)),
            ("lr", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ])),
    ]


def evaluate(X, y, class_names, groups, label):
    print(f"\n--- {label} ---")
    print(f"  classes  : {class_names}")
    print(f"  counts   : {pd.Series(y).value_counts().sort_index().to_dict()}")
    print(f"  random F1: {1.0 / len(class_names):.3f}")

    rows = []
    for name, factory in make_candidates():
        mean, folds = loso_macro_f1(factory, X, y, groups)
        rows.append({"model": name, "mean_f1": mean, "folds": folds})
    df = pd.DataFrame(rows)
    df["folds"] = df["folds"].apply(lambda fs: "[" + ", ".join(f"{f:.3f}" for f in fs) + "]")
    print(df.to_string(index=False))
    best = max(rows, key=lambda r: r["mean_f1"])
    above_random = best["mean_f1"] - 1.0 / len(class_names)
    print(f"  best: {best['model']}  -> macro-F1 = {best['mean_f1']:.3f} "
          f"(+{above_random:.3f} above random)")
    return best


def main():
    X_raw = np.load(PROCESSED_DIR / "stage2" / "X.npy")
    meta = pd.read_parquet(PROCESSED_DIR / "stage2" / "meta.parquet")
    X, names = extract_features_batch(X_raw)
    groups = meta["subject"].values

    # ----- 5-class setup -----
    class_names_5 = sorted(meta["class_name"].unique())
    y_5 = meta["class_name"].map({n: i for i, n in enumerate(class_names_5)}).values
    best_5 = evaluate(X, y_5, class_names_5, groups, "5-class setup")

    # ----- 3-class setup -----
    merged = meta["class_name"].map(MERGE_MAP)
    class_names_3 = sorted(merged.unique())
    y_3 = merged.map({n: i for i, n in enumerate(class_names_3)}).values
    best_3 = evaluate(X, y_3, class_names_3, groups, "3-class setup")

    # ----- side-by-side ----
    print("\n" + "=" * 60)
    print("Comparison (best classifier per setup):")
    print(f"  5-class:  macro-F1 = {best_5['mean_f1']:.3f}  ({best_5['model']})  "
          f"  baseline = 0.200  -> +{best_5['mean_f1']-0.20:.3f}")
    print(f"  3-class:  macro-F1 = {best_3['mean_f1']:.3f}  ({best_3['model']})  "
          f"  baseline = 0.333  -> +{best_3['mean_f1']-1.0/3:.3f}")
    print()
    print("Interpretation:")
    print("  Compare the +delta values, NOT the raw F1.  A 3-class model has a")
    print("  higher baseline so its raw F1 will naturally look better.")
    print("  Only if delta-above-random is meaningfully higher does the 3-class")
    print("  setup actually capture more signal.")


if __name__ == "__main__":
    main()
