import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import io

# Helper Functions for Image Modifier
def adjust_brightness(image, factor):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

def adjust_contrast(image, factor):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def flip_image(image, mode):
    if mode == "Horizontal":
        return cv2.flip(image, 1)
    elif mode == "Vertical":
        return cv2.flip(image, 0)
    return image

# Helper Functions for Image Compressor
def compress_image(image, quality, format_type):
    buffer = io.BytesIO()
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image.save(buffer, format=format_type, quality=quality)
    buffer.seek(0)
    return buffer

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Image Modifier Page
def modify_image():
    st.title("🔧 Advanced Image Modifier")
    st.write("Upload an image and explore advanced enhancement options.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display Original Image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

        # User Enhancement Options
        enhancements = st.multiselect(
            "Select Enhancements",
            ["Grayscale", "Sketch", "Blur", "Canny Edges", "Brightness", "Contrast", "Rotate", "Flip"]
        )

        processed_image = image

        if "Grayscale" in enhancements:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        
        if "Sketch" in enhancements:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            inverted_gray = 255 - gray
            blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
            inverted_blur = 255 - blurred
            processed_image = cv2.divide(gray, inverted_blur, scale=256.0)
        
        if "Blur" in enhancements:
            ksize = st.slider("Select Blur Intensity", 1, 50, 5, step=2)
            processed_image = cv2.GaussianBlur(processed_image, (ksize, ksize), 0)
        
        if "Canny Edges" in enhancements:
            threshold1 = st.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.slider("Threshold 2", 0, 255, 200)
            processed_image = cv2.Canny(processed_image, threshold1, threshold2)
        
        if "Brightness" in enhancements:
            brightness_factor = st.slider("Adjust Brightness", 0.5, 2.0, 1.0)
            processed_image = adjust_brightness(processed_image, brightness_factor)
        
        if "Contrast" in enhancements:
            contrast_factor = st.slider("Adjust Contrast", 0.5, 2.0, 1.0)
            processed_image = adjust_contrast(processed_image, contrast_factor)
        
        if "Rotate" in enhancements:
            angle = st.slider("Rotate Angle (degrees)", -180, 180, 0)
            processed_image = rotate_image(processed_image, angle)
        
        if "Flip" in enhancements:
            flip_mode = st.radio("Flip Mode", ["None", "Horizontal", "Vertical"])
            if flip_mode != "None":
                processed_image = flip_image(processed_image, flip_mode)

        # Display Processed Image
        st.subheader("Processed Image")
        if processed_image.ndim == 2:  # Grayscale image
            st.image(processed_image, channels="GRAY")
        else:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), channels="RGB")

        # Download Button
        result_image = Image.fromarray(
            processed_image if processed_image.ndim == 2
            else cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        )
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            label="Download Processed Image",
            data=buffer,
            file_name="processed_image.png",
            mime="image/png",
        )

# Image Compressor Page
def compress_image_section():
    st.title("🗜️ Image Compressor")
    st.write("Upload an image and compress it.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Read the image
        image = Image.open(uploaded_file)

        # Show Original Image
        st.subheader("Original Image")
        st.image(image)

        # Compression Settings
        quality = st.slider("Compression Quality (%)", 10, 100, 85)
        resize_option = st.checkbox("Resize Image Before Compression?")
        if resize_option:
            width = st.slider("Width", 10, 1000, 500)
            height = st.slider("Height", 10, 1000, 500)
            image = resize_image(np.array(image), width, height)

        # Select file format
        format_type = st.selectbox("Choose File Format for Compression", ["JPEG", "PNG"])

        # Compress Image
        compressed_image = compress_image(image, quality, format_type)

        # Show Download Button
        st.download_button(
            label="Download Compressed Image",
            data=compressed_image,
            file_name=f"compressed_image.{format_type.lower()}",
            mime=f"image/{format_type.lower()}",
        )

# Navigation
def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to:", ["Image Modifier", "Image Compressor"])

    if selected_page == "Image Modifier":
        modify_image()
    elif selected_page == "Image Compressor":
        compress_image_section()

if __name__ == "__main__":
    main()
