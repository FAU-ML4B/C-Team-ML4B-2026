"""
End-to-end inference for the Streamlit demo.

Public function: predict(folder_path) -> dict

Interface kept identical to Leopold's earlier stub so the existing
Streamlit app continues to work without changes; only the returned
values switch from hard-coded mock data to real model predictions.

Returned dict:
  fall       : bool     -- did Stage 1 classify the window as a fall?
  type       : str|None -- predicted fall direction (None if not a fall)
  confidence : float    -- Stage 1 probability of the fall class
  peak_g     : float    -- peak acceleration magnitude in g
  severity   : str      -- coarse impact bucket ("none", "mild", "moderate", "severe")

The function loads the trained models lazily on first call and caches
them at module level. Streamlit can additionally use @st.cache_resource
around its own wrapper to keep them warm across reruns.
"""
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np

from src.parsing import parse_recording
from src.preprocessing import (
    apply_butterworth_filter,
    peak_triggered_window,
    signal_magnitude,
    G_TO_MS2,
)
from src.features import extract_features_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Loaded lazily on first call.
_stage1 = None
_stage2 = None


def _load_models() -> None:
    """Load both .joblib bundles. Raises FileNotFoundError with a helpful
    message if the models have not been trained yet."""
    global _stage1, _stage2
    if _stage1 is not None and _stage2 is not None:
        return

    stage1_path = MODELS_DIR / "stage1_binary.joblib"
    stage2_path = MODELS_DIR / "stage2_falltype.joblib"
    if not stage1_path.exists() or not stage2_path.exists():
        raise FileNotFoundError(
            f"Trained models missing. Expected {stage1_path} and {stage2_path}. "
            f"Run `python -m src.train` first."
        )
    _stage1 = joblib.load(stage1_path)
    _stage2 = joblib.load(stage2_path)


def _severity_from_peak_g(peak_g: float) -> str:
    """Coarse, paper-justifiable severity buckets based on peak |a|.

    Thresholds derived from common fall-detection literature:
      < 1.8 g  -> no significant impact
      1.8-3 g  -> mild (low fall, soft surface)
      3-6 g   -> moderate (typical mattress fall)
      >= 6 g   -> severe (hard surface, high impact)
    """
    if peak_g < 1.8:
        return "none"
    if peak_g < 3.0:
        return "mild"
    if peak_g < 6.0:
        return "moderate"
    return "severe"


def predict(folder_path: Union[str, Path]) -> dict:
    """Run the two-stage classifier on a single Sensor Logger recording.

    Parameters
    ----------
    folder_path : str | Path
        Path to an unzipped Sensor Logger recording folder
        (must contain Accelerometer.csv, Gyroscope.csv, Gravity.csv).

    Returns
    -------
    dict with keys: fall, type, confidence, peak_g, severity.
    """
    folder = Path(folder_path)
    _load_models()

    # 1. Parse and filter the recording
    df = parse_recording(folder)
    if df is None:
        return {
            "fall": False,
            "type": None,
            "confidence": 0.0,
            "peak_g": 0.0,
            "severity": "none",
        }
    df_f = apply_butterworth_filter(df)

    # 2. Peak |a| in g across the whole recording (always reported)
    svm = signal_magnitude(df_f, sensor="accel")
    if np.median(svm) > 5.0:
        svm = svm / G_TO_MS2  # convert m/s^2 -> g if needed
    peak_g = float(np.max(svm))

    # 3. Try to extract a peak-aligned window; if it's too quiet, no fall.
    window = peak_triggered_window(df_f)
    if window is None:
        return {
            "fall": False,
            "type": None,
            "confidence": 0.0,
            "peak_g": peak_g,
            "severity": _severity_from_peak_g(peak_g),
        }

    # 4. Feature extraction
    X, _ = extract_features_batch(window[None, ...])

    # 5. Stage-1: binary fall vs ADL
    stage1_model = _stage1["model"]
    proba_fall = float(stage1_model.predict_proba(X)[0, 1])
    is_fall = bool(stage1_model.predict(X)[0])

    if not is_fall:
        return {
            "fall": False,
            "type": None,
            "confidence": proba_fall,
            "peak_g": peak_g,
            "severity": _severity_from_peak_g(peak_g),
        }

    # 6. Stage-2: 5-class fall type
    stage2_model = _stage2["model"]
    class_names = _stage2["class_names"]
    pred_idx = int(stage2_model.predict(X)[0])
    fall_type = class_names[pred_idx]

    return {
        "fall": True,
        "type": fall_type,
        "confidence": proba_fall,
        "peak_g": peak_g,
        "severity": _severity_from_peak_g(peak_g),
    }
