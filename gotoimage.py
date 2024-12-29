import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Helper functions for effects
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blur = 255 - blurred
    return cv2.divide(gray, inverted_blur, scale=256.0)

def apply_blur(image, ksize):
    ksize = max(1, ksize)
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_canny_edges(image, threshold1, threshold2):
    return cv2.Canny(image, threshold1, threshold2)

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(image, kernel)

def adjust_brightness(image, beta):
    return cv2.convertScaleAbs(image, beta=beta)

def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Main Streamlit app
def main():
    st.title("Advanced Image Modification App 🎨")
    st.write("Upload an image, apply effects, customize controls, and download the result!")

    # File uploader
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

        # Effect selection
        st.subheader("Choose an Effect")
        effects = [
            "None", "Grayscale", "Sketch", "Blur", "Canny Edges",
            "Sepia", "Brightness Adjustment", "Contrast Adjustment",
            "Rotate", "Flip", "Resize"
        ]
        effect = st.selectbox("Effect", effects)

        # Additional customization options
        if effect == "Blur":
            ksize = st.slider("Blur Intensity", 1, 50, 5, step=2)
            processed_image = apply_blur(image, ksize)
        elif effect == "Canny Edges":
            threshold1 = st.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.slider("Threshold 2", 0, 255, 200)
            processed_image = apply_canny_edges(image, threshold1, threshold2)
        elif effect == "Sepia":
            processed_image = apply_sepia(image)
        elif effect == "Brightness Adjustment":
            beta = st.slider("Brightness Level", -100, 100, 0)
            processed_image = adjust_brightness(image, beta)
        elif effect == "Contrast Adjustment":
            alpha = st.slider("Contrast Level", 0.5, 3.0, 1.0)
            processed_image = adjust_contrast(image, alpha)
        elif effect == "Rotate":
            angle = st.slider("Rotation Angle", -180, 180, 0)
            processed_image = rotate_image(image, angle)
        elif effect == "Flip":
            flip_options = {"Horizontal": 1, "Vertical": 0, "Both": -1}
            flip_direction = st.selectbox("Flip Direction", list(flip_options.keys()))
            processed_image = flip_image(image, flip_options[flip_direction])
        elif effect == "Resize":
            width = st.number_input("Width", min_value=10, value=image.shape[1])
            height = st.number_input("Height", min_value=10, value=image.shape[0])
            if st.button("Resize"):
                processed_image = resize_image(image, int(width), int(height))
            else:
                processed_image = image
        elif effect == "Grayscale":
            processed_image = convert_to_grayscale(image)
        elif effect == "Sketch":
            processed_image = convert_to_sketch(image)
        else:
            processed_image = image

        # Display processed image
        st.subheader("Processed Image")
        if effect in ["Grayscale", "Sketch", "Canny Edges"]:
            st.image(processed_image, channels="GRAY")
        else:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), channels="RGB")

        # Download button
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
            mime="image/png"
        )

if __name__ == "__main__":
    main()
