import streamlit as st
import tempfile
import zipfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference import predict


def find_recording_folder(tmp_path: Path) -> Path:
    """
    Find the actual Sensor Logger recording folder after ZIP extraction.

    Sensor Logger ZIP files often contain one top-level folder with files like:
    Accelerometer.csv, Gyroscope.csv, Gravity.csv.

    The outer temp folder itself usually does not contain these CSV files directly.
    Therefore, we search for the folder that contains Accelerometer.csv.
    """
    ignored_folders = {"__MACOSX"}

    for folder in tmp_path.rglob("*"):
        if not folder.is_dir():
            continue

        if any(part in ignored_folders for part in folder.parts):
            continue

        if (folder / "Accelerometer.csv").exists():
            return folder

    return tmp_path


LOGO_PATH = str(Path(__file__).resolve().parent / "assets" / "safestep_logo.png")

st.set_page_config(
    page_title="SafeStep – Fall Detection",
    page_icon=LOGO_PATH,
    layout="wide"
)

st.image(LOGO_PATH, width=320)

st.write(
    "Upload a Sensor Logger ZIP file to detect whether a fall occurred "
    "and estimate the fall type and impact intensity."
)

st.info(
    "This app shows three possible detection states: no fall, uncertain, and fall. "
    "Low-confidence positive predictions are shown as uncertain instead of confirmed falls."
)

uploaded_file = st.file_uploader(
    "Upload Sensor Logger ZIP",
    type=["zip"]
)

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        zip_path = tmp_path / uploaded_file.name
        zip_path.write_bytes(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_path)

        st.success("File uploaded and extracted successfully.")

        recording_path = find_recording_folder(tmp_path)
        result = predict(recording_path)

        confidence = result.get("confidence", 0.0)
        is_fall = result.get("fall", False)
        fall_type = result.get("type")
        peak_g = result.get("peak_g", 0.0)
        severity = result.get("severity", "unknown")

        st.subheader("Prediction result")

        if is_fall and confidence < 0.75:
            st.warning(
                f"⚠️ Uncertain detection "
                f"(confidence {confidence:.0%}).\n\n"
                f"Peak acceleration {peak_g:.2f} g detected, "
                f"but the system cannot reliably classify this as a fall."
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Detection state", "UNCERTAIN")
            col2.metric("Fall type", "—")
            col3.metric("Confidence", f"{confidence:.0%}")
            col4.metric("Peak acceleration", f"{peak_g:.2f} g")

        elif is_fall:
            shown_type = fall_type if fall_type else "unknown"

            st.error(
                f"🚨 Fall detected — {shown_type} "
                f"({confidence:.0%} confidence)"
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Detection state", "FALL")
            col2.metric("Fall type", shown_type)
            col3.metric("Confidence", f"{confidence:.0%}")
            col4.metric("Peak acceleration", f"{peak_g:.2f} g")

            st.metric("Severity", severity)

        else:
            st.success(
                f"✅ No fall detected "
                f"(confidence {1 - confidence:.0%})"
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Detection state", "NO FALL")
            col2.metric("Fall type", "—")
            col3.metric("Confidence", f"{1 - confidence:.0%}")
            col4.metric("Peak acceleration", f"{peak_g:.2f} g")

        st.info(
            "Note: The primary output is binary fall detection. "
            "Fall-type classification is experimental. "
            "Low-confidence positive predictions are shown as uncertain to avoid "
            "overconfident fall alerts for high-acceleration activities such as running or jumping."
        )

        with st.expander("Raw prediction output"):
            st.json(result)

        with st.expander("Extracted files"):
            files = [
                str(p.relative_to(tmp_path))
                for p in tmp_path.rglob("*")
                if p.is_file()
            ]
            st.write(files)

        with st.expander("Recording folder used for prediction"):
            st.write(str(recording_path.relative_to(tmp_path)))

else:
    st.info("Please upload a Sensor Logger ZIP file to start.")
