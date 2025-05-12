import streamlit as st
from utils.yolo_utils import load_model, preprocess_image, predict_and_draw
import torch

st.set_page_config(page_title="Flying Object Detector", layout="centered")
st.title("🛩️ Flying Object Detector (YOLO + EfficientNet)")
st.markdown("Upload an image of a flying object (drone, plane, etc.) and detect what it is.")

# Load model once and cache
@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("model/efficientnet_yolo.pt", device=device)
    return model, device

model, device = get_model()

# Upload UI
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil, _ = preprocess_image(uploaded_file)
    st.image(image_pil, caption="Original Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Detecting..."):
            output_image, detected_labels = predict_and_draw(image_pil, model, device)
            st.image(output_image, caption="Prediction", use_column_width=True)

            if detected_labels:
                st.markdown("### ✅ Detected Objects:")
                for label in detected_labels:
                    st.write(f"- {label}")
            else:
                st.warning("No objects detected with the current threshold.")



