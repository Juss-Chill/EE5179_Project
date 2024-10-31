mport streamlit as st
from PIL import Image
import numpy as np


# Dummy mask generation function (replace this with your actual mask function)
def generate_mask(image):
    # Convert image to grayscale and threshold it to create a binary mask for demonstration
    grayscale_image = image.convert("L")  # Convert to grayscale
    mask_array = np.array(grayscale_image) > 128  # Thresholding
    mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))  # Convert back to Image
    return mask_image


# Streamlit UI
st.title(":green[Image Mask Generator]")

# Upload Image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display the original image
    st.subheader("Original Image")
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate the mask
    mask_image = generate_mask(image)

    # Display the mask image
    st.subheader("Denoise Image")
    st.image(mask_image, caption="Generated Mask", use_column_width=True)