import streamlit as st
from PIL import Image
import io
import base64
from visual_style_transfer import run_style_transfer, image_loader

# Streamlit page config
st.set_page_config(page_title="🎨 Visual Style Transfer", layout="centered")
st.title("🎨 Visual Style Transfer using PyTorch")
st.markdown("Upload a **content image** and a **style image**, then watch them blend into art!")

# Sidebar uploaders
with st.sidebar:
    st.header("Upload Images")
    content_file = st.file_uploader("📷 Content Image", type=["jpg", "png", "jpeg"])
    style_file = st.file_uploader("🖼️ Style Image", type=["jpg", "png", "jpeg"])
    run_button = st.button("✨ Stylize")

# Function to generate fullscreen HTML view
def get_fullscreen_view_button(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_img = base64.b64encode(img_bytes).decode()

    # Wrap image in an HTML page for fullscreen feel
    html_page = f'''
        <html>
            <body style="margin:0;">
                <img src="data:image/png;base64,{base64_img}" 
                     style="width:100%;height:auto;display:block;" />
            </body>
        </html>
    '''
    encoded_html = base64.b64encode(html_page.encode()).decode()

    # Create a button that opens that HTML page in a new tab
    html = f'''
        <a target="_blank" href="data:text/html;base64,{encoded_html}">
            <button style="font-size:18px;padding:10px 20px;margin-top:10px;">🔍 View Fullscreen</button>
        </a>
    '''
    return html

# Run style transfer if both files are uploaded
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

    # Add the fullscreen viewer button
    st.markdown(get_fullscreen_view_button(output_img), unsafe_allow_html=True)

elif not content_file or not style_file:
    st.info("⬅️ Please upload both content and style images from the sidebar and click 'Stylize'.")

