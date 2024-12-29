import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Helper Functions for Image Modifier
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blur = 255 - blurred
    return cv2.divide(gray, inverted_blur, scale=256.0)

def apply_blur(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_canny_edges(image, threshold1, threshold2):
    return cv2.Canny(image, threshold1, threshold2)

# Helper Functions for Image Compressor
def compress_image(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer

# Home Page UI
def home_page():
    st.title("🎨 Advanced Image Tool Suite 🌟")
    st.subheader("Enhance and Compress Images with Ease")

    col1, col2 = st.columns(2)

    with col1:
        st.header("🔧 Image Modifier")
        st.write("Apply effects and transformations to your images.")
        if st.button("Go to Image Modifier"):
            st.session_state.page = "modifier"

    with col2:
        st.header("🗜️ Image Compressor")
        st.write("Compress images and reduce file size.")
        if st.button("Go to Image Compressor"):
            st.session_state.page = "compressor"

# Image Modifier Page
def modify_image():
    st.title("🔧 Image Modifier")
    st.write("Upload an image and apply effects.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Show Original Image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

        # Choose Effect
        effect = st.selectbox("Choose an Effect", ["None", "Grayscale", "Sketch", "Blur", "Canny Edges"])
        if effect == "Grayscale":
            processed_image = convert_to_grayscale(image)
        elif effect == "Sketch":
            processed_image = convert_to_sketch(image)
        elif effect == "Blur":
            ksize = st.slider("Blur Intensity", 1, 50, 5, step=2)
            processed_image = apply_blur(image, ksize)
        elif effect == "Canny Edges":
            threshold1 = st.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.slider("Threshold 2", 0, 255, 200)
            processed_image = apply_canny_edges(image, threshold1, threshold2)
        else:
            processed_image = image

        # Show Processed Image
        st.subheader("Processed Image")
        if effect in ["Grayscale", "Sketch", "Canny Edges"]:
            st.image(processed_image, channels="GRAY")
        else:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), channels="RGB")

        # Download Button
        result_image = Image.fromarray(
            processed_image if effect in ["Grayscale", "Sketch", "Canny Edges"]
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
    
    # Back to Home Button
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"

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

        # Compression Slider
        quality = st.slider("Compression Quality (%)", 10, 100, 85)

        # Compress Image
        compressed_image = compress_image(image, quality)

        # Show Download Button
        st.download_button(
            label="Download Compressed Image",
            data=compressed_image,
            file_name="compressed_image.jpg",
            mime="image/jpeg",
        )
    
    # Back to Home Button
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"

# Page Navigation
def navigate():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Home", "Image Modifier", "Image Compressor"], key="sidebar_navigation")

    # Sidebar navigation
    if st.session_state.sidebar_navigation == "Home":
        st.session_state.page = "home"
    elif st.session_state.sidebar_navigation == "Image Modifier":
        st.session_state.page = "modifier"
    elif st.session_state.sidebar_navigation == "Image Compressor":
        st.session_state.page = "compressor"

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "modifier":
        modify_image()
    elif st.session_state.page == "compressor":
        compress_image_section()

if __name__ == "__main__":
    navigate()
