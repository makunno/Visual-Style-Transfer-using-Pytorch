from PIL import Image
import streamlit as st
from visual_style_transfer import run_style_transfer, image_loader

st.title("🎨 Visual Style Transfer")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

if content_file and style_file:
    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    with st.spinner("Generating style-transferred image..."):
        output_img = run_style_transfer(
            image_loader(content_img),
            image_loader(style_img)
        )

    st.image(output_img, caption="Styled Output")

