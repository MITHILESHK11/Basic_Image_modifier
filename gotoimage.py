import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Effect functions
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    inverted_blur = 255 - blurred
    return cv2.divide(gray, inverted_blur, scale=256.0)

def invert_colors(image):
    return cv2.bitwise_not(image)

def apply_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def apply_canny_edges(image):
    return cv2.Canny(image, 100, 200)

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(image, kernel)

def adjust_brightness(image, beta=50):
    return cv2.convertScaleAbs(image, beta=beta)

def adjust_contrast(image, alpha=1.5):
    return cv2.convertScaleAbs(image, alpha=alpha)

def cartoonify(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

# Main Streamlit app
def main():
    st.title("Comprehensive Image Modification App 🎨")
    st.write("Upload an image, choose from a wide range of effects, and download the result!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

        # Effect selection
        effects = [
            "Grayscale", "Sketch", "Invert Colors", "Blur",
            "Canny Edges", "Sepia", "Brightness Adjustment",
            "Contrast Adjustment", "Cartoonify"
        ]
        effect = st.selectbox("Choose an effect", effects)

        # Process image based on effect
        if effect == "Grayscale":
            processed_image = convert_to_grayscale(image)
        elif effect == "Sketch":
            processed_image = convert_to_sketch(image)
        elif effect == "Invert Colors":
            processed_image = invert_colors(image)
        elif effect == "Blur":
            processed_image = apply_blur(image)
        elif effect == "Canny Edges":
            processed_image = apply_canny_edges(image)
        elif effect == "Sepia":
            processed_image = apply_sepia(image)
        elif effect == "Brightness Adjustment":
            processed_image = adjust_brightness(image, beta=50)
        elif effect == "Contrast Adjustment":
            processed_image = adjust_contrast(image, alpha=1.5)
        elif effect == "Cartoonify":
            processed_image = cartoonify(image)
        else:
            st.warning("Effect not implemented!")

        # Display processed image
        st.subheader("Modified Image")
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
            label="Download Modified Image",
            data=buffer,
            file_name="modified_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
