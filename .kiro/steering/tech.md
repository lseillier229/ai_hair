---
inclusion: always
---

# Tech Stack

## Core Technologies

- **Python 3.x** - Primary language
- **Streamlit** - Web UI framework for interactive apps
- **OpenCV (opencv-python-headless)** - Image processing
- **MediaPipe** - Face mesh detection and selfie segmentation
- **NumPy** - Numerical operations
- **PyAV** - Video frame handling
- **ONNX Runtime** - Optional ML model inference (BiSeNet)

## Key Libraries

- `streamlit-webrtc` - Real-time camera streaming in browser
- `mediapipe` - Google's ML solutions for face landmarks
- `onnxruntime` - Cross-platform ML inference
- `Pillow (PIL)` - Image loading and manipulation

## Common Commands

### Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```

### Development
The app runs on `http://localhost:8501` by default. Streamlit auto-reloads on file changes.

## Model Files

BiSeNet ONNX model should be placed at `models/bisenet_faceparsing.onnx` (optional, configurable via UI). The `models/` directory is gitignored.
