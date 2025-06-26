import streamlit as st
from PIL import Image
import io
from visual_style_transfer import run_style_transfer  # Replace with your actual import

st.set_page_config(page_title="Visual Style Transfer", layout="centered")

st.title("🎨 Visual Style Transfer")
st.markdown("Upload a **content image** and a **style image**, and see the magic!")

with st.sidebar:
    st.header("Upload Images")
    content_file = st.file_uploader("📷 Content Image", type=["png", "jpg", "jpeg"])
    style_file = st.file_uploader("🖼️ Style Image", type=["png", "jpg", "jpeg"])
    run_button = st.button("✨ Stylize")

if content_file and style_file and run_button:
    content_img = Image.open(content_file).convert('RGB')
    style_img = Image.open(style_file).convert('RGB')

    with st.spinner("Running style transfer..."):
        output_img = run_style_transfer(content_img, style_img)

    st.subheader("🔍 Result")
    st.image(output_img, caption="Stylized Output", use_column_width=True)

    # Optional: Download button
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="💾 Download Stylized Image",
        data=byte_im,
        file_name="stylized_output.png",
        mime="image/png"
    )
else:
    st.info("Upload both images and click 'Stylize' to begin.")

