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


st.set_page_config(
    page_title="Smartphone Fall Detection",
    page_icon="📱",
    layout="wide"
)

st.title("Smartphone-Based Fall Detection")
st.write(
    "Upload a Sensor Logger ZIP file to detect whether a fall occurred "
    "and estimate the fall type and impact intensity."
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

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Fall detected?", "YES" if result["fall"] else "NO")
        col2.metric("Fall type", result["type"] if result["type"] else "—")
        col3.metric("Confidence", f"{result['confidence']:.0%}")
        col4.metric("Peak acceleration", f"{result['peak_g']:.2f} g")

        st.subheader("Prediction result")
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
