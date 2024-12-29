# 🎨 Image Modifier and Compressor App

## Overview

This web-based application allows users to upload images and perform a variety of operations such as enhancements, transformations, and compression. Built using **Streamlit**, **OpenCV**, and **Pillow**, it provides a user-friendly interface to modify and compress images with real-time previews and adjustable parameters. The app is suitable for casual users looking to enhance photos, as well as developers or researchers who need quick and intuitive image processing capabilities.

---

## Features

### 🔧 **Image Modifier**
- **Grayscale**: Converts the uploaded image to black-and-white.
- **Sketch Effect**: Simulates a pencil sketch of the image using edge detection and inversion.
- **Blur Effect**: Smooths the image with a user-controlled Gaussian blur intensity.
- **Canny Edge Detection**: Highlights edges in the image for a stylized or analytical look.
- **Brightness Adjustment**: Fine-tunes image brightness.
- **Contrast Adjustment**: Enhances or reduces image contrast.
- **Rotate**: Rotates the image to a user-defined angle (clockwise or counterclockwise).
- **Flip**: Flips the image either horizontally or vertically.

### 🗜️ **Image Compressor**
- **Compression**: Reduce the file size by lowering quality settings.
- **Resize Option**: Resize images to specific dimensions (width and height).
- **Format Conversion**: Choose between JPEG and PNG output formats.

### 🖼️ **Preview and Download**
- Preview the modified or compressed image directly in the app.
- Download the processed image with a single click.

---

## Try the App

🌐 **[Try the App Here](https://hotelchatbot-79v9xgaadawlkgzoqoycwg.streamlit.app/)**  

---

## Requirements

### Prerequisites
1. **Python 3.8 or higher**
2. Libraries:
   - **Streamlit** (for the web interface)
   - **OpenCV** (for image processing)
   - **Pillow** (for handling image enhancements and formats)
   - **NumPy** (for numerical array operations)

Install all dependencies using pip:

```bash
pip install streamlit opencv-python pillow numpy
