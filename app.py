import streamlit as st
from PIL import Image
import io
import base64
from visual_style_transfer import run_style_transfer, image_loader

# Streamlit page config
st.set_page_config(page_title="🎨 Visual Style Transfer", layout="centered")
st.title("🎨 Visual Style Transfer using PyTorch")
st.markdown("Upload a **content image** and a **style image**, then watch them merge into art!")

# Sidebar uploaders
with st.sidebar:
    st.header("Upload Images")
    content_file = st.file_uploader("📷 Content Image", type=["jpg", "png", "jpeg"])
    style_file = st.file_uploader("🖼️ Style Image", type=["jpg", "png", "jpeg"])
    run_button = st.button("✨ Stylize")

# Run style transfer if images are uploaded
if content_file and style_file and run_button:
    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    with st.spinner("🔧 Running style transfer... please wait."):
        output_img = run_style_transfer(
            image_loader(content_img),
            image_loader(style_img)
        )

    st.subheader("🔍 Stylized Output")
    st.image(output_img, caption="Styled Output", use_column_width=True)

    # Function to generate download/view link
    def get_image_download_link(img, filename):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/png;base64,{img_str}" target="_blank" download="{filename}">🔍 Open Full Image</a>'
        return href

    # Add clickable open/download link
    st.markdown(get_image_download_link(output_img, "stylized_output.png"), unsafe_allow_html=True)

elif not content_file or not style_file:
    st.info("⬅️ Please upload both content and style images from the sidebar and click 'Stylize'.")


