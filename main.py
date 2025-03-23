import streamlit as st
import numpy as np
import cv2
import os
import random
import io
from PIL import Image
from pillow_lut import load_cube_file  # Load LUT as Pillow filter

# Define LUT folder and grain file path
LUT_FOLDER = "luts"  # Change this to your LUT folder path
GRAIN_FILE = "grain.jpg"  # Change this to your grain file path

# Function to list all LUTs
def list_luts():
    return [f for f in os.listdir(LUT_FOLDER) if f.endswith(".cube")]

# Function to blend images using "Overlay" mode
def blend_overlay(base, overlay):
    base = base.astype(np.float32) / 255.0
    overlay = overlay.astype(np.float32) / 255.0
    result = np.where(base < 0.5, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay))
    return (result * 255).astype(np.uint8)

# Function to apply grain overlay
def apply_grain_overlay(image, grain_file):
    # Load the large grain texture
    grain = cv2.imread(grain_file, cv2.IMREAD_GRAYSCALE)
    h_grain, w_grain = grain.shape

    # Randomly crop a 6000x4000 section
    x_start = random.randint(0, w_grain - 6000)
    y_start = random.randint(0, h_grain - 4000)
    grain_crop = grain[y_start:y_start + 4000, x_start:x_start + 6000]

    # Resize grain to match image while keeping aspect ratio
    h_img, w_img = image.shape[:2]
    aspect_ratio = grain_crop.shape[1] / grain_crop.shape[0]

    if w_img / h_img > aspect_ratio:
        new_w = w_img
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = h_img
        new_w = int(new_h * aspect_ratio)

    grain_resized = cv2.resize(grain_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop to match the exact image dimensions
    start_x = (new_w - w_img) // 2
    start_y = (new_h - h_img) // 2
    grain_final = grain_resized[start_y:start_y + h_img, start_x:start_x + w_img]

    # Convert grayscale grain to 3 channels
    grain_final = cv2.cvtColor(grain_final, cv2.COLOR_GRAY2BGR)

    # Apply overlay blend mode
    return blend_overlay(image, grain_final)

# Streamlit UI
st.title("LUT & Grain Overlay Processor")

# LUT Selection
luts = list_luts()
if not luts:
    st.error("No LUTs found in the LUT folder.")
else:
    selected_lut = st.selectbox("Select a LUT", luts)

# Image Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image_src = Image.open(uploaded_file).convert("RGB")

    # Show the original image
    st.image(image_src, caption="Original Image", use_column_width=True)

    # Process button
    if st.button("Compute"):
        # Load and apply LUT
        lut_path = os.path.join(LUT_FOLDER, selected_lut)
        lut = load_cube_file(lut_path)
        image_lut = image_src.filter(lut)  # Apply LUT using Pillow filter

        # Convert PIL to NumPy for grain processing
        image_lut_np = np.array(image_lut)

        # Apply grain overlay
        final_image_np = apply_grain_overlay(image_lut_np, GRAIN_FILE)

        # Convert NumPy array back to PIL Image
        final_image = Image.fromarray(final_image_np)

        # **FIX**: Save image to a buffer before downloading
        img_buffer = io.BytesIO()
        final_image.save(img_buffer, format="JPEG")  # Save as JPEG
        img_buffer.seek(0)  # Reset buffer position

        # Show the processed image
        st.image(final_image, caption="Processed Image", use_column_width=True)

        # Offer download with properly formatted image
        st.download_button(
            label="Download Processed Image",
            data=img_buffer,
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )
