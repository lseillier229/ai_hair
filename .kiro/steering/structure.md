---
inclusion: always
---

# Project Structure

## Root Files

- `app.py` - Main Streamlit application (single-file architecture)
- `requirements.txt` - Python dependencies
- `styles_catalog_example.json` - Example hairstyle catalog structure

## Directories

- `models/` - ONNX model files (gitignored, not committed)
- `venv/` - Python virtual environment (gitignored)
- `.kiro/` - Kiro AI assistant configuration and steering rules

## Architecture Pattern

**Single-file Streamlit app** - All logic in `app.py`:
- MediaPipe initialization at module level
- Helper functions for distance calculation and face shape classification
- `BiseNetHairONNX` class for optional ONNX inference
- `LiveTransformer` class (extends `VideoTransformerBase`) for camera mode
- UI controls in sidebar
- Two main execution paths: camera mode and image upload mode

## Code Organization in app.py

1. **Imports and setup** - Libraries and Streamlit page config
2. **MediaPipe initialization** - Face mesh, selfie segmentation, drawing utils
3. **Landmark indices** - Dictionary mapping facial features to landmark IDs
4. **Helper functions** - `dist()`, `face_shape_from_landmarks()`, `overlay_mask()`
5. **BiseNetHairONNX class** - Optional ONNX model wrapper
6. **UI controls** - Sidebar configuration
7. **LiveTransformer class** - Real-time video processing
8. **Mode routing** - Camera vs image upload logic

## Face Shape Classification

Uses 8 key landmarks: chin, forehead, left/right zygomatic (cheekbones), left/right jaw, left/right temples. Classification based on ratios:
- `cheek_to_length` - Face width vs length
- `jaw_to_cheek` - Jaw width vs cheekbone width  
- `temples_to_cheek` - Temple width vs cheekbone width

## Styles Catalog Format

JSON structure with hairstyle metadata:
- `style_id` - Unique identifier
- `name` - Display name
- `tags` - Face shapes, length, texture, fringe compatibility
- `assets` - Image paths for overlays and references
