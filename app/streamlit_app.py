import base64
import tempfile
import zipfile
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference import predict


APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "safestep_logo.png"

PAGE_ICON = str(LOGO_PATH) if LOGO_PATH.exists() else "📱"

st.set_page_config(
    page_title="SafeStep · Fall Detection",
    page_icon=PAGE_ICON,
    layout="wide",
)


def image_to_base64(path: Path) -> str | None:
    """Convert image file to base64 for embedding in custom HTML."""
    if not path.exists():
        return None

    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


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


def render_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f5faff 0%, #ffffff 45%, #f7fbff 100%);
            }

            .block-container {
                padding-top: 0rem;
                padding-left: 4rem;
                padding-right: 4rem;
                max-width: 1280px;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            .top-header {
                background: #071d33;
                margin-left: -4rem;
                margin-right: -4rem;
                padding: 1.15rem 4rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 4px solid #1d9bf0;
            }

            .brand-area {
                display: flex;
                align-items: center;
                gap: 1rem;
            }

            .logo-img {
                height: 58px;
                width: auto;
                object-fit: contain;
            }

            .logo-fallback {
                font-size: 1.95rem;
                font-weight: 800;
                letter-spacing: -0.04em;
                color: #ffffff;
            }

            .header-right {
                color: #8fc9ff;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.14em;
            }

            .hero-section {
                display: grid;
                grid-template-columns: 1fr 260px;
                gap: 2rem;
                align-items: center;
                padding: 3.2rem 0 2.2rem 0;
            }

            .eyebrow {
                color: #0b6fb8;
                font-size: 0.78rem;
                font-weight: 900;
                letter-spacing: 0.16em;
                margin-bottom: 0.8rem;
            }

            .hero-section h1 {
                font-size: 3.2rem;
                line-height: 1.02;
                margin: 0;
                letter-spacing: -0.055em;
                color: #071d33;
            }

            .hero-section p {
                color: #4d6172;
                font-size: 1.05rem;
                line-height: 1.55;
                max-width: 760px;
                margin-top: 1rem;
            }

            .status-card {
                background: #ffffff;
                border: 1px solid #d8e8f7;
                border-radius: 18px;
                padding: 1.25rem 1.35rem;
                box-shadow: 0 16px 42px rgba(7, 29, 51, 0.08);
            }

            .status-label {
                color: #7a91a6;
                font-size: 0.7rem;
                font-weight: 900;
                letter-spacing: 0.14em;
                margin-bottom: 0.6rem;
            }

            .status-dot-row {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: #071d33;
                font-weight: 800;
            }

            .status-dot {
                width: 10px;
                height: 10px;
                background: #23c483;
                border-radius: 50%;
                display: inline-block;
                box-shadow: 0 0 0 5px rgba(35, 196, 131, 0.12);
            }

            .section-card {
                background: #ffffff;
                border: 1px solid #dceaf7;
                border-radius: 22px;
                padding: 1.35rem;
                box-shadow: 0 18px 44px rgba(7, 29, 51, 0.07);
                min-height: 370px;
                margin-bottom: 1.5rem;
            }

            .section-title {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                color: #071d33;
                font-weight: 900;
                font-size: 0.95rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 1.1rem;
                padding-bottom: 0.85rem;
                border-bottom: 1px solid #e1edf8;
            }

            .step-badge {
                background: #071d33;
                color: #ffffff;
                width: 28px;
                height: 28px;
                border-radius: 9px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 0.85rem;
                font-weight: 900;
            }

            .upload-help {
                color: #5f7386;
                font-size: 0.92rem;
                line-height: 1.45;
                margin-bottom: 1rem;
            }

            [data-testid="stFileUploader"] {
                background: #f7fbff;
                border: 1px dashed #b8d8f2;
                border-radius: 18px;
                padding: 1.25rem;
            }

            [data-testid="stFileUploader"] section {
                background: transparent;
                border: none;
            }

            .result-card {
                border-radius: 22px;
                padding: 1.35rem 1.5rem;
                margin-top: 1.2rem;
                margin-bottom: 1.1rem;
                border: 1px solid;
            }

            .result-card h2 {
                margin: 0.15rem 0 0.5rem 0;
                font-size: 1.65rem;
                letter-spacing: -0.03em;
            }

            .result-card p {
                margin: 0;
                line-height: 1.45;
            }

            .result-topline {
                font-size: 0.74rem;
                font-weight: 900;
                letter-spacing: 0.15em;
                text-transform: uppercase;
            }

            .result-fall {
                background: #fff3f3;
                border-color: #ffc9c9;
                color: #8a1010;
            }

            .result-uncertain {
                background: #fff8e8;
                border-color: #ffe1a6;
                color: #7a4c00;
            }

            .result-nofall {
                background: #eefaf4;
                border-color: #bfeed4;
                color: #0b5d36;
            }

            .metric-card {
                background: #ffffff;
                border: 1px solid #dceaf7;
                border-radius: 18px;
                padding: 1.1rem 1rem;
                box-shadow: 0 12px 30px rgba(7, 29, 51, 0.055);
                min-height: 108px;
                margin-bottom: 0.8rem;
            }

            .metric-label {
                color: #6f8498;
                font-size: 0.72rem;
                text-transform: uppercase;
                font-weight: 900;
                letter-spacing: 0.1em;
                margin-bottom: 0.55rem;
            }

            .metric-value {
                color: #071d33;
                font-size: 1.45rem;
                font-weight: 900;
                letter-spacing: -0.03em;
            }

            .severity-card {
                background: #f7fbff;
                border-left: 4px solid #1d9bf0;
                border-radius: 12px;
                padding: 0.85rem 1rem;
                margin-top: 1rem;
                color: #253b4f;
            }

            .note-card {
                background: #eef6ff;
                border: 1px solid #cde6fb;
                border-radius: 18px;
                padding: 1rem 1.15rem;
                color: #24435e;
                margin-top: 1.2rem;
                line-height: 1.45;
            }

            .waiting-card {
                background: #ffffff;
                border: 1px solid #dceaf7;
                border-radius: 22px;
                padding: 2rem;
                text-align: center;
                box-shadow: 0 18px 44px rgba(7, 29, 51, 0.07);
                color: #4d6172;
            }

            .waiting-icon {
                font-size: 2.6rem;
                margin-bottom: 0.8rem;
            }

            .footer-line {
                margin-top: 2rem;
                padding-top: 1rem;
                border-top: 1px solid #dceaf7;
                color: #8398ab;
                font-size: 0.8rem;
                display: flex;
                justify-content: space-between;
            }

            @media (max-width: 900px) {
                .hero-section {
                    grid-template-columns: 1fr;
                }

                .block-container {
                    padding-left: 1.25rem;
                    padding-right: 1.25rem;
                }

                .top-header {
                    margin-left: -1.25rem;
                    margin-right: -1.25rem;
                    padding-left: 1.25rem;
                    padding-right: 1.25rem;
                }

                .hero-section h1 {
                    font-size: 2.35rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render custom SafeStep header."""
    logo_base64 = image_to_base64(LOGO_PATH)

    if logo_base64:
        logo_html = f"""
        <img src="data:image/png;base64,{logo_base64}" class="logo-img">
        """
    else:
        logo_html = """
        <div class="logo-fallback">SafeStep</div>
        """

    st.markdown(
        f"""
        <div class="top-header">
            <div class="brand-area">
                {logo_html}
            </div>
            <div class="header-right">
                SMARTPHONE FALL DETECTION
            </div>
        </div>

        <div class="hero-section">
            <div>
                <div class="eyebrow">ML4B · SENSOR-BASED SAFETY INTELLIGENCE</div>
                <h1>Stürze sichtbar machen.</h1>
                <p>
                    Upload einer Sensor-Logger-ZIP-Datei, automatische Analyse der
                    Smartphone-IMU-Daten und verständliche Ausgabe als
                    <strong>no fall</strong>, <strong>uncertain</strong> oder <strong>fall</strong>.
                </p>
            </div>
            <div class="status-card">
                <div class="status-label">SYSTEMSTATUS</div>
                <div class="status-dot-row">
                    <span class="status-dot"></span>
                    Bereit für Analyse
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(number: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <span class="step-badge">{number}</span>
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(
    detection_state: str,
    title: str,
    description: str,
    confidence_label: str,
    peak_g: float,
    fall_type: str,
    severity: str,
    state_class: str,
) -> None:
    """Render custom result card."""
    st.markdown(
        f"""
        <div class="result-card {state_class}">
            <div class="result-topline">{detection_state}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Detection State</div>
                <div class="metric-value">{detection_state}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Fall Type</div>
                <div class="metric-value">{fall_type}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Peak Acceleration</div>
                <div class="metric-value">{peak_g:.2f} g</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if severity != "—":
        st.markdown(
            f"""
            <div class="severity-card">
                <strong>Impact severity:</strong> {severity}
            </div>
            """,
            unsafe_allow_html=True,
        )


render_css()
render_header()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_header("1", "Datenerfassung")

    st.markdown(
        """
        <div class="upload-help">
            Lade eine <strong>Sensor Logger ZIP-Datei</strong> hoch.
            Die App entpackt die Aufnahme automatisch und sucht den passenden Recording-Folder.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Sensor Logger ZIP",
        type=["zip"],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="note-card">
            Erwartete Dateien im Recording-Folder:
            <strong>Accelerometer.csv</strong>, optional weitere Sensor-Dateien wie
            Gyroscope.csv oder Gravity.csv.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_header("2", "Befund & Interpretation")

    result_placeholder = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        zip_path = tmp_path / uploaded_file.name
        zip_path.write_bytes(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_path)

        recording_path = find_recording_folder(tmp_path)
        result = predict(recording_path)

        confidence = result.get("confidence", 0.0)
        is_fall = result.get("fall", False)
        fall_type = result.get("type")
        peak_g = result.get("peak_g", 0.0)
        severity = result.get("severity", "unknown")

        if is_fall and confidence < 0.75:
            detection_state = "UNCERTAIN"
            title = "Uncertain detection"
            description = (
                f"Peak acceleration detected, but confidence is only "
                f"{confidence:.0%}. The system does not communicate this "
                f"as a confirmed fall."
            )
            confidence_label = f"{confidence:.0%}"
            shown_type = "—"
            shown_severity = "—"
            state_class = "result-uncertain"

        elif is_fall:
            detection_state = "FALL"
            shown_type = fall_type if fall_type else "unknown"
            title = "Fall detected"
            description = (
                f"The model detected a fall-like movement with "
                f"{confidence:.0%} confidence."
            )
            confidence_label = f"{confidence:.0%}"
            shown_severity = severity
            state_class = "result-fall"

        else:
            detection_state = "NO FALL"
            title = "No fall detected"
            description = (
                "The uploaded recording was classified as a normal "
                "or non-fall movement."
            )
            confidence_label = f"{1 - confidence:.0%}"
            shown_type = "—"
            shown_severity = "—"
            state_class = "result-nofall"

        with result_placeholder.container():
            render_result_card(
                detection_state=detection_state,
                title=title,
                description=description,
                confidence_label=confidence_label,
                peak_g=peak_g,
                fall_type=shown_type,
                severity=shown_severity,
                state_class=state_class,
            )

            st.markdown(
                """
                <div class="note-card">
                    <strong>Interpretation:</strong>
                    The primary output is binary fall detection. Fall-type classification
                    is experimental. Low-confidence positive predictions are shown as
                    uncertain to avoid overconfident fall alerts for high-acceleration
                    activities such as running or jumping.
                </div>
                """,
                unsafe_allow_html=True,
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
    with result_placeholder.container():
        st.markdown(
            """
            <div class="waiting-card">
                <div class="waiting-icon">📱</div>
                <h3>Bereit für den ersten Befund</h3>
                <p>
                    Nach dem Upload erscheinen hier Detection State,
                    Confidence, Peak Acceleration und optional der Falltyp.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class="footer-line">
        <span>SafeStep · Smartphone-Based Fall Detection</span>
        <span>ML4B · C-Team</span>
    </div>
    """,
    unsafe_allow_html=True,
)
