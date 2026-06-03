"""
Hand-crafted feature extraction for IMU windows.

Each input window has shape (n_samples, n_channels). The function
extracts ~70 features per window and returns a flat feature vector.

Feature groups:
  - Per-axis time-domain stats (mean, std, min, max, range, RMS, skew, kurt)
  - Signal-vector magnitude (SVM) stats on accel and gyro
  - Jerk (1st derivative of accel) peak and integral
  - Frequency-domain energy in low / mid / high bands via FFT

Channel order is fixed:
  [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
"""
from typing import List

import numpy as np
from scipy import stats
from scipy.fft import rfft

SAMPLE_RATE_HZ = 50.0
CHANNEL_NAMES = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")
FREQ_BANDS_HZ = ((0.0, 3.0), (3.0, 6.0), (6.0, 10.0))


def _stats_1d(x: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_skew": float(stats.skew(x)) if np.std(x) > 1e-9 else 0.0,
        f"{prefix}_kurt": float(stats.kurtosis(x)) if np.std(x) > 1e-9 else 0.0,
    }


def _band_energy(x: np.ndarray, prefix: str) -> dict:
    spectrum = np.abs(rfft(x - np.mean(x)))
    freqs = np.linspace(0, SAMPLE_RATE_HZ / 2, len(spectrum))
    out = {}
    for lo, hi in FREQ_BANDS_HZ:
        mask = (freqs >= lo) & (freqs < hi)
        out[f"{prefix}_energy_{int(lo)}_{int(hi)}Hz"] = float(np.sum(spectrum[mask] ** 2))
    return out


def extract_features_window(window: np.ndarray) -> dict:
    """Extract all features for one window (n_samples, 6)."""
    feats: dict = {}

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        feats.update(_stats_1d(window[:, ch_idx], ch_name))

    accel_svm = np.linalg.norm(window[:, :3], axis=1)
    gyro_svm = np.linalg.norm(window[:, 3:], axis=1)
    feats.update(_stats_1d(accel_svm, "accel_svm"))
    feats.update(_stats_1d(gyro_svm, "gyro_svm"))

    jerk = np.diff(accel_svm) * SAMPLE_RATE_HZ
    feats["jerk_peak"] = float(np.max(np.abs(jerk)))
    feats["jerk_integral"] = float(np.sum(np.abs(jerk)) / SAMPLE_RATE_HZ)
    feats["jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))

    feats.update(_band_energy(accel_svm, "accel_svm"))

    return feats


def extract_features_batch(windows: np.ndarray) -> "tuple[np.ndarray, List[str]]":
    """Apply extract_features_window to every window in a batch."""
    feat_dicts = [extract_features_window(w) for w in windows]
    names = list(feat_dicts[0].keys())
    X = np.array([[d[name] for name in names] for d in feat_dicts], dtype=np.float64)
    return X, names
