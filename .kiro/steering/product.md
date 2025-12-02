---
inclusion: always
---

# Product Overview

This is a face analysis and hairstyle recommendation application built with Streamlit. The app provides:

- **Face shape detection** using MediaPipe Face Mesh (468 landmarks)
- **Hair/person segmentation** using BiSeNet ONNX model (optional) or MediaPipe Selfie Segmentation (fallback)
- **Two modes**: Live camera feed and static image upload
- **Visual overlay** showing detected face shape and segmentation mask

The application classifies faces into shapes: oval, round, oblong, square, heart, and diamond based on facial landmark ratios. It's designed for hairstyle recommendations based on face geometry.

The app supports a styles catalog (JSON format) for matching hairstyles to face shapes, though this is currently a reference structure.
