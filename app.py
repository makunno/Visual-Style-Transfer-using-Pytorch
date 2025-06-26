import streamlit as st
from PIL import Image
import io
import base64
from visual_style_transfer import run_style_transfer, image_loader

# Streamlit page config
st.set_page_config(page_title="🎨 Visual Style Transfer", layout="centered")
st.title("🎨 Visual Style Transfer using PyTorch")
st.markdown("Upload a **content image** and a **style image**, then watch them blend into art!")

# Sidebar for uploads
with st.sidebar:
    st.header("Upload Images")
    content_file = st.file_uploader("📷 Content Image", type=["jpg", "png", "jpeg"])
    style_file = st.file_uploader("🖼️ Style Image", type=["jpg", "png", "jpeg"])
    run_button = st.button("✨ Stylize")

# Function to generate "View Full Image" button
def get_image_view_link(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()

    html = f'''
    <a href="data:image/png;base64,{img_base64}" target="_blank">
        <button style="font-size:18px;padding:10px 20px;margin-top:10px;">🔍 View Full Image</button>
    </a>
    '''
    return html

# Run style transfer if both images are uploaded
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

    # Show View Full Image button
    st.markdown(get_image_view_link(output_img), unsafe_allow_html=True)

elif not content_file or not style_file:
    st.info("⬅️ Please upload both content and style images from the sidebar and click 'Stylize'.")

