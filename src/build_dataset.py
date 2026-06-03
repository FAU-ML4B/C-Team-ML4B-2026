"""
End-to-end dataset builder, v2.

Key change vs v1:
  For Stage-1 (fall vs ADL), each FALL recording contributes EXACTLY ONE
  positive sample - the peak-aligned 3 s window centred on the impact.
  ADL recordings contribute their full set of sliding windows.
  This eliminates the label noise that crippled v1, where every 2 s chunk
  of a fall recording (including the 5 s of standing still beforehand and
  the 5 s of lying still afterwards) was labelled `fall`.

Produces two datasets:

1. Stage-1 dataset (data/processed/stage1/)
   - Falls : peak-aligned 3 s windows  (one per fall recording)
   - ADL   : sliding 2 s windows       (many per ADL recording)
   - Both standardised to (150 samples, 6 channels) by zero-padding the
     ADL windows on both sides so Stage-1 sees a uniform input shape.

2. Stage-2 dataset (data/processed/stage2/)
   - Peak-aligned 3 s windows from FALL recordings only, with the
     5 fall-class labels.

Usage:
    python -m src.build_dataset data/raw data/processed
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.parsing import parse_recording, read_metadata
from src.preprocessing import (
    apply_butterworth_filter,
    parse_label_from_folder,
    sliding_windows,
    peak_triggered_window,
    iter_recording_folders,
    PEAK_PRE_SAMPLES,
    PEAK_POST_SAMPLES,
    SLIDING_WINDOW_SAMPLES,
)

FALL_CLASSES = {"backwards", "forward", "left", "right", "stumble"}
PEAK_WINDOW_LEN = PEAK_PRE_SAMPLES + PEAK_POST_SAMPLES   # 150 samples


def _pad_sliding_to_peak_shape(window: np.ndarray) -> np.ndarray:
    """Zero-pad a 100-sample sliding window to 150 samples so Stage-1
    sees a uniform input length across falls (150) and ADL (originally 100).
    Padding is symmetric (25 zeros front, 25 zeros back).
    """
    pad = (PEAK_WINDOW_LEN - SLIDING_WINDOW_SAMPLES) // 2
    n_channels = window.shape[1]
    out = np.zeros((PEAK_WINDOW_LEN, n_channels), dtype=window.dtype)
    out[pad:pad + window.shape[0]] = window
    return out


def build(data_raw: Path, data_processed: Path) -> None:
    data_raw = Path(data_raw)
    data_processed = Path(data_processed)
    (data_processed / "stage1").mkdir(parents=True, exist_ok=True)
    (data_processed / "stage2").mkdir(parents=True, exist_ok=True)

    stage1_X, stage1_meta = [], []   # binary fall/adl
    stage2_X, stage2_meta = [], []   # 5-class fall type

    skipped_parse = 0
    skipped_label = 0
    falls_without_peak = 0
    falls_with_peak = 0
    standardisation_off = 0

    folders = list(iter_recording_folders(data_raw))
    print(f"Found {len(folders)} recording folders under {data_raw}\n")

    for folder in folders:
        try:
            label = parse_label_from_folder(folder.name)
        except ValueError as e:
            print(f"  ! Label parse failed: {folder.name} ({e})")
            skipped_label += 1
            continue

        df = parse_recording(folder)
        if df is None:
            print(f"  ! Parse/quality check failed: {folder.name}")
            skipped_parse += 1
            continue

        if str(read_metadata(folder).get("standardisation", "")).lower() == "false":
            standardisation_off += 1

        df_f = apply_butterworth_filter(df)
        is_fall = label["class_name"] in FALL_CLASSES

        if is_fall:
            # One peak-aligned window per fall recording, for BOTH stages.
            peak_w = peak_triggered_window(df_f)
            if peak_w is None:
                falls_without_peak += 1
                print(f"  ! Fall without detectable peak (skipped): {folder.name}")
                continue
            falls_with_peak += 1
            stage1_X.append(peak_w)
            stage1_meta.append({**label, "binary_label": 1, "source": "peak"})
            stage2_X.append(peak_w)
            stage2_meta.append(label)
        else:
            # ADL: sliding windows, all labelled 0 (= adl)
            sw = sliding_windows(df_f)
            for w_idx, window in enumerate(sw):
                stage1_X.append(_pad_sliding_to_peak_shape(window))
                stage1_meta.append(
                    {**label, "binary_label": 0, "source": "sliding", "window_idx": w_idx}
                )

    # ---- Save stage1 dataset ----
    if stage1_X:
        s1_X = np.array(stage1_X)
        s1_meta = pd.DataFrame(stage1_meta)
        np.save(data_processed / "stage1" / "X.npy", s1_X)
        s1_meta.to_parquet(data_processed / "stage1" / "meta.parquet")
        print(f"\nStage1 dataset: X.shape = {s1_X.shape}")
        print(f"  positives (fall): {(s1_meta['binary_label'] == 1).sum()}")
        print(f"  negatives (adl) : {(s1_meta['binary_label'] == 0).sum()}")

    # ---- Save stage2 dataset ----
    if stage2_X:
        s2_X = np.array(stage2_X)
        s2_meta = pd.DataFrame(stage2_meta)
        np.save(data_processed / "stage2" / "X.npy", s2_X)
        s2_meta.to_parquet(data_processed / "stage2" / "meta.parquet")
        print(f"Stage2 dataset: X.shape = {s2_X.shape}")

    # ---- Summary ----
    print(f"\nSummary:")
    print(f"  recordings found        : {len(folders)}")
    print(f"  skipped (bad label)     : {skipped_label}")
    print(f"  skipped (parse/quality) : {skipped_parse}")
    print(f"  falls with peak (kept)  : {falls_with_peak}")
    print(f"  falls without peak      : {falls_without_peak}")
    print(f"  standardisation = False : {standardisation_off}")

    if stage1_X:
        s1_meta = pd.DataFrame(stage1_meta)
        print(f"\nStage1 fall vs ADL per subject:")
        pivot = (s1_meta.assign(label=s1_meta["binary_label"].map({0: "adl", 1: "fall"}))
                 .groupby(["subject", "label"])
                 .size().unstack(fill_value=0))
        print(pivot)

    if stage2_X:
        s2_meta = pd.DataFrame(stage2_meta)
        print(f"\nStage2 fall-type per subject:")
        print(s2_meta.groupby(["subject", "class_name"]).size().unstack(fill_value=0))


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.build_dataset <data_raw_dir> <data_processed_dir>")
        sys.exit(1)
    build(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
