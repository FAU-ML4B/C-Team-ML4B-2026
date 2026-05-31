"""
End-to-end dataset builder: data/raw/ -> data/processed/

Walks every recording folder under data/raw/, parses each (parsing.py),
filters and windows it (preprocessing.py), then saves windowed arrays
plus a metadata table to data/processed/.

Produces two datasets so each downstream classifier gets the right input:

1. Sliding-window dataset (data/processed/sliding/)
   - Every 2-second chunk of every recording becomes one sample.
   - Used for the binary fall-vs-ADL classifier.

2. Peak-aligned dataset (data/processed/peak/)
   - Exactly one 3-second window per fall recording, centred on the impact.
   - Used for the fall-type classifier.

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
)


def build(data_raw: Path, data_processed: Path) -> None:
    data_raw = Path(data_raw)
    data_processed = Path(data_processed)
    (data_processed / "sliding").mkdir(parents=True, exist_ok=True)
    (data_processed / "peak").mkdir(parents=True, exist_ok=True)

    sliding_X, sliding_meta = [], []
    peak_X, peak_meta = [], []

    skipped_parse = 0
    skipped_label = 0
    skipped_peak = 0
    standardisation_off = 0

    folders = list(iter_recording_folders(data_raw))
    print(f"Found {len(folders)} recording folders under {data_raw}\n")

    for folder in folders:
        # 1. Label first - if the folder name violates the schema, skip
        #    the recording entirely so bad data never reaches the model.
        try:
            label = parse_label_from_folder(folder.name)
        except ValueError as e:
            print(f"  ! Label parse failed: {folder.name} ({e})")
            skipped_label += 1
            continue

        # 2. Parse + resample
        df = parse_recording(folder)
        if df is None:
            print(f"  ! Parse/quality check failed: {folder.name}")
            skipped_parse += 1
            continue

        # 3. Note recordings made with standardisation off (informational)
        meta = read_metadata(folder)
        if str(meta.get("standardisation", "")).lower() == "false":
            standardisation_off += 1

        # 4. Filter
        df_f = apply_butterworth_filter(df)

        # 5. Sliding windows for every recording
        windows = sliding_windows(df_f)
        for w_idx, window in enumerate(windows):
            sliding_X.append(window)
            sliding_meta.append({**label, "window_idx": w_idx})

        # 6. Peak-triggered window: only meaningful for fall classes.
        #    ADL classes won't (normally) cross the 1.8 g threshold,
        #    so peak_triggered_window naturally returns None and we skip.
        peak_window = peak_triggered_window(df_f)
        if peak_window is None:
            skipped_peak += 1
        else:
            peak_X.append(peak_window)
            peak_meta.append(label)

    # ---- Save sliding dataset ----
    if sliding_X:
        sliding_X_arr = np.array(sliding_X)
        sliding_meta_df = pd.DataFrame(sliding_meta)
        np.save(data_processed / "sliding" / "X.npy", sliding_X_arr)
        sliding_meta_df.to_parquet(data_processed / "sliding" / "meta.parquet")
        print(f"\nSliding dataset:  X.shape = {sliding_X_arr.shape}")
    else:
        print("\nNo sliding windows produced.")

    # ---- Save peak dataset ----
    if peak_X:
        peak_X_arr = np.array(peak_X)
        peak_meta_df = pd.DataFrame(peak_meta)
        np.save(data_processed / "peak" / "X.npy", peak_X_arr)
        peak_meta_df.to_parquet(data_processed / "peak" / "meta.parquet")
        print(f"Peak dataset:     X.shape = {peak_X_arr.shape}")
    else:
        print("No peak windows produced.")

    # ---- Summary ----
    print(f"\nSummary:")
    print(f"  recordings found        : {len(folders)}")
    print(f"  skipped (bad label)     : {skipped_label}")
    print(f"  skipped (parse/quality) : {skipped_parse}")
    print(f"  no peak above threshold : {skipped_peak}")
    print(f"  standardisation = False : {standardisation_off}")

    if sliding_X:
        class_counts = (
            pd.DataFrame(sliding_meta)
            .groupby(["subject", "class_name"])
            .size()
            .unstack(fill_value=0)
        )
        print(f"\nSliding windows per subject/class:\n{class_counts}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.build_dataset <data_raw_dir> <data_processed_dir>")
        sys.exit(1)
    build(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
