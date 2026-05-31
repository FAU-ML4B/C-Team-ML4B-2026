"""
Filter, windowing, and label extraction.

Refactored from Lea's prepare_data.ipynb (Cells 3 & 4) with two key changes:

1. Labels are extracted from the recording folder name following the agreed
   schema {subject}_{placement}_{class}_{trial}_{date}, instead of using
   d.parent.name (which previously gave a unique-per-recording label including
   timestamp, which would have broken the classifier).

2. The Butterworth filter is applied to all motion sensor channels
   (accel + gyro), not just the accelerometer.

The two windowing strategies are exposed as separate functions:
- sliding_windows: for binary fall-vs-ADL classification
- peak_triggered_window: for fall-type classification on the impact moment
"""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import signal

# ----- Filter constants -----
FILTER_ORDER = 4
FILTER_CUTOFF_HZ = 15.0
SAMPLE_RATE_HZ = 50.0

# ----- Sliding window constants (binary fall vs ADL) -----
SLIDING_WINDOW_SAMPLES = 100  # 2.0 s at 50 Hz
SLIDING_STEP_SAMPLES = 50     # 50 % overlap

# ----- Peak-triggered window constants (fall type) -----
PEAK_THRESHOLD_G = 1.8
PEAK_PRE_SAMPLES = 50    # 1.0 s before the peak
PEAK_POST_SAMPLES = 100  # 2.0 s after the peak
G_TO_MS2 = 9.80665

# ----- Label schema -----
LABEL_FIELDS = ("subject", "placement", "class_name", "trial", "date")


def apply_butterworth_filter(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Apply a 4th-order low-pass Butterworth filter to motion-sensor columns.

    Defaults to all columns starting with accel_ or gyro_ if not specified.
    Returns a new DataFrame.
    """
    if columns is None:
        columns = [c for c in df.columns if c.startswith(("accel_", "gyro_"))]
    if not columns:
        return df.copy()

    df = df.copy()
    b, a = signal.butter(
        FILTER_ORDER, FILTER_CUTOFF_HZ, btype="low", fs=SAMPLE_RATE_HZ
    )
    for col in columns:
        df[col] = signal.filtfilt(b, a, df[col].values)
    return df


def parse_label_from_folder(folder_name: str) -> dict:
    """Extract structured label from a folder name following the schema:
        {subject}_{placement}_{class}_{trial}_{date}
    e.g.  felix_pocket_stumble_03_20260523

    Raises ValueError if the schema is violated, so a bad folder name
    surfaces immediately rather than silently producing wrong labels.
    """
    parts = folder_name.split("_")
    if len(parts) != 5:
        raise ValueError(
            f"Folder name '{folder_name}' does not match schema "
            f"subject_placement_class_trial_date "
            f"(got {len(parts)} parts, expected 5)"
        )
    subject, placement, class_name, trial, date = parts
    return {
        "subject": subject,
        "placement": placement,
        "class_name": class_name,
        "trial": trial,
        "date": date,
        "folder_name": folder_name,
    }


def sliding_windows(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    window_samples: int = SLIDING_WINDOW_SAMPLES,
    step: int = SLIDING_STEP_SAMPLES,
) -> np.ndarray:
    """Cut a recording into overlapping fixed-size windows.

    Returns array of shape (n_windows, window_samples, n_channels).
    Used for binary fall-vs-ADL classification where the model sees
    every chunk of the recording independently.
    """
    if columns is None:
        columns = [c for c in df.columns if c.startswith(("accel_", "gyro_"))]
    data = df[list(columns)].values
    n_samples = len(data)

    windows = []
    for start in range(0, n_samples - window_samples + 1, step):
        windows.append(data[start:start + window_samples])
    return np.array(windows)


def signal_magnitude(df: pd.DataFrame, sensor: str = "accel") -> np.ndarray:
    """Compute the magnitude |a| = sqrt(x^2+y^2+z^2) of a 3-axis sensor."""
    x = df[f"{sensor}_x"].values
    y = df[f"{sensor}_y"].values
    z = df[f"{sensor}_z"].values
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def peak_triggered_window(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    threshold_g: float = PEAK_THRESHOLD_G,
    pre: int = PEAK_PRE_SAMPLES,
    post: int = PEAK_POST_SAMPLES,
) -> Optional[np.ndarray]:
    """Find the impact peak and return a 3-second window aligned to it.

    The peak is detected on |a| of the accelerometer (auto-normalised
    to g if values look like m/s^2). Returns None if no peak above
    threshold or if the peak is too close to the start/end of the
    recording to extract the full window.

    Output shape: (pre + post, n_channels).
    """
    if columns is None:
        columns = [c for c in df.columns if c.startswith(("accel_", "gyro_"))]

    svm = signal_magnitude(df, sensor="accel")
    # Normalise to g if the values look like m/s^2 (median > ~5 g would be
    # unreasonable for a phone in a pocket).
    if np.median(svm) > 5.0:
        svm = svm / G_TO_MS2

    if np.max(svm) < threshold_g:
        return None

    peak_idx = int(np.argmax(svm))
    if peak_idx - pre < 0 or peak_idx + post > len(df):
        return None

    data = df[list(columns)].values
    return data[peak_idx - pre:peak_idx + post]


def iter_recording_folders(data_dir: Path):
    """Yield every recording folder under data_dir (any folder that
    contains an Accelerometer.csv).
    """
    data_dir = Path(data_dir)
    for csv in sorted(data_dir.rglob("Accelerometer.csv")):
        yield csv.parent
