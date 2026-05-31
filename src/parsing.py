"""
Sensor Logger recording folder -> resampled multi-sensor DataFrame.

Refactored from Lea's prepare_data.ipynb (Cell 1), with:
- Multi-sensor support (Accelerometer, Gyroscope, Gravity)
- Label extraction decoupled (see src/labeling.py)
- Type hints and module-level constants for transparency
- Quality checks return None instead of raising, so a bad recording
  is dropped silently rather than crashing the batch pipeline

Tested against iPhone iOS Sensor Logger 1.59 recordings sampled at 100 Hz.
"""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Constants - kept module-level so they appear in any future review
SENSORS_DEFAULT: Sequence[str] = ("Accelerometer", "Gyroscope", "Gravity")
SENSOR_PREFIX = {
    "Accelerometer": "accel",
    "Gyroscope": "gyro",
    "Gravity": "grav",
    "Orientation": "orient",
}
TARGET_RATE_HZ = 50
RESAMPLE_PERIOD = f"{int(1000 / TARGET_RATE_HZ)}ms"  # "20ms"
MAX_MEDIAN_GAP_S = 0.025  # drop recordings whose median sample gap > 25 ms
MIN_ROWS = 20             # need at least this many samples per sensor


def _load_sensor_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load one Sensor Logger CSV, drop pre-Start rows, run quality check.

    Returns the cleaned DataFrame, or None if quality checks fail.
    """
    df = pd.read_csv(csv_path)

    # Sensor Logger sometimes buffers samples from BEFORE the user hit Start;
    # those have negative seconds_elapsed and should be dropped.
    df = df[df["seconds_elapsed"] >= 0].copy()
    if len(df) < MIN_ROWS:
        return None

    # Quality check: median sample gap. If gaps exceed 25 ms, effective
    # rate is below 40 Hz - the recording is unreliable.
    gaps = df["seconds_elapsed"].diff().dropna()
    if gaps.median() > MAX_MEDIAN_GAP_S:
        return None

    return df


def _resample_sensor(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Resample a single sensor to TARGET_RATE_HZ on a datetime index.

    Renames the x/y/z columns with the sensor prefix so multiple sensors
    can be merged without collision (e.g. accel_x vs gyro_x).
    """
    df = df.copy()
    df.index = pd.to_datetime(df["time"], unit="ns")

    cols = ["x", "y", "z"]
    df_rs = df[cols].resample(RESAMPLE_PERIOD).mean().interpolate("linear")
    df_rs = df_rs.rename(columns={c: f"{prefix}_{c}" for c in cols})
    return df_rs


def parse_recording(
    folder: Path,
    sensors: Sequence[str] = SENSORS_DEFAULT,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """Parse one Sensor Logger recording folder into a unified DataFrame.

    Each requested sensor CSV is loaded, quality-checked, resampled to 50 Hz,
    and merged on a shared datetime index. Rows where any sensor has NaN
    after merging are dropped.

    Parameters
    ----------
    folder : Path
        Path to the recording folder containing the individual sensor CSVs.
    sensors : Sequence[str]
        Sensor names to load. Defaults to Accelerometer, Gyroscope, Gravity.
    verbose : bool
        Print warnings to stdout when sensors are skipped.

    Returns
    -------
    DataFrame with columns like ['accel_x','accel_y','accel_z','gyro_x',...]
    indexed by datetime at 50 Hz, or None if no sensor could be loaded.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Recording folder not found: {folder}")

    sensor_dfs = []
    for sensor_name in sensors:
        csv_path = folder / f"{sensor_name}.csv"
        if not csv_path.exists():
            if verbose:
                print(f"  - {sensor_name}.csv missing in {folder.name}")
            continue

        raw = _load_sensor_csv(csv_path)
        if raw is None:
            if verbose:
                print(f"  - {sensor_name} quality check failed in {folder.name}")
            continue

        prefix = SENSOR_PREFIX.get(sensor_name, sensor_name.lower()[:5])
        sensor_dfs.append(_resample_sensor(raw, prefix))

    if not sensor_dfs:
        return None

    # Merge sensors on the datetime index, drop rows with NaN in any column.
    merged = pd.concat(sensor_dfs, axis=1).dropna()
    if len(merged) < MIN_ROWS:
        return None

    return merged


def read_metadata(folder: Path) -> dict:
    """Read Metadata.csv from a recording folder and return as a dict.

    Useful for sanity checks - especially the 'standardisation' flag,
    which should be 'true' for cross-platform consistency.
    """
    folder = Path(folder)
    meta_path = folder / "Metadata.csv"
    if not meta_path.exists():
        return {}

    df = pd.read_csv(meta_path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()
