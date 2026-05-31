"""Smoke tests for src.parsing on real Sensor Logger recordings.

Run from project root:
    python tests/test_parsing.py path/to/recording_folder

Or to tests multiple recordings at once:
    python tests/test_parsing.py path/to/parent_folder_with_recordings
"""
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsing import parse_recording, read_metadata, TARGET_RATE_HZ


def smoke_test_recording(folder: Path) -> bool:
    """Parse one recording and print a short summary. Returns True if OK."""
    print(f"\n--- {folder.name} ---")

    # 1. Metadata sanity
    meta = read_metadata(folder)
    if meta:
        std = meta.get("standardisation")
        device = meta.get("device name")
        rate = meta.get("sampleRateMs")
        print(f"  device         : {device}")
        print(f"  sampleRateMs   : {rate}")
        print(f"  standardisation: {std}")
        if str(std).lower() == "false":
            print("  ! WARNING: 'Standardise Units & Frames' was OFF for this recording.")
    else:
        print("  ! Metadata.csv missing or empty.")

    # 2. Parse
    df = parse_recording(folder, verbose=True)
    if df is None:
        print("  X parse_recording returned None - recording dropped.")
        return False

    # 3. Sanity-check the output
    n_rows, n_cols = df.shape
    duration_s = (df.index[-1] - df.index[0]).total_seconds()
    effective_rate = n_rows / duration_s if duration_s > 0 else float("nan")
    print(f"  rows           : {n_rows}")
    print(f"  cols           : {list(df.columns)}")
    print(f"  duration       : {duration_s:.2f} s")
    print(f"  effective rate : {effective_rate:.1f} Hz (target {TARGET_RATE_HZ})")
    print(f"  first 2 rows   :\n{df.head(2)}")

    ok = (n_rows > 50) and (abs(effective_rate - TARGET_RATE_HZ) < 5)
    print(f"  -> {'OK' if ok else 'FAILED checks'}")
    return ok


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Path does not exist: {target}")
        sys.exit(1)

    # Find all candidate recording folders: any folder containing Accelerometer.csv
    if (target / "Accelerometer.csv").exists():
        folders = [target]
    else:
        folders = sorted(p.parent for p in target.rglob("Accelerometer.csv"))

    if not folders:
        print(f"No recordings (folders with Accelerometer.csv) found under {target}")
        sys.exit(1)

    print(f"Found {len(folders)} recording(s).")
    results = [smoke_test_recording(f) for f in folders]
    n_ok = sum(results)
    print(f"\n=== {n_ok}/{len(results)} recordings passed smoke tests ===")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
