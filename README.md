# Smartphone-Based Fall Detection

This project is a prototype for smartphone-based fall detection using IMU sensor data. The system is designed to process Sensor Logger ZIP files, extract motion data, and display fall detection results in a Streamlit web application (https://c-team-ml4b-fall-detection.streamlit.app).

## Project Goal

The goal is to detect whether a fall occurred based on smartphone accelerometer and gyroscope data. If a fall is detected, the system should also estimate the fall type and the impact intensity.

Planned fall types:

- trip-stumble
- forward
- left side 
- right side
- backwards

Planned non-fall activities:

- walking
- sitting
- stairs
- jumping

## Current Prototype

The current Streamlit prototype supports:

- uploading a Sensor Logger ZIP file
- extracting the uploaded ZIP file
- calling a prediction function via `src.inference.predict()`
- displaying prediction results in the web app

At the moment, the prediction function is a mock function. It returns fixed example values and will later be replaced by the real machine learning inference pipeline.

## Project Structure

```text
ml4b-falldetection/
├── app/
│   └── streamlit_app.py
├── docs/
│   ├── related_work.md
│   └── presentation_outline.md
├── src/
│   └── inference.py
├── project.md
├── README.md
└── requirements.txt
