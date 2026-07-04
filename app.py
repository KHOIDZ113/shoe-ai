import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ================= UI CONFIG =================
st.set_page_config(
    page_title="Shoe AI Classifier",
    page_icon="👟",
    layout="centered"
)

# ================= HEADER =================
st.markdown(
    """
    <h1 style='text-align:center; color:#4F8BF9;'>👟 Shoe AI Classifier</h1>
    <p style='text-align:center;'>Upload ảnh giày và AI sẽ dự đoán loại giày</p>
    """,
    unsafe_allow_html=True
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("shoe_classifier.h5")

model = load_model()

# ================= LOAD LABELS =================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ================= UPLOAD =================
file = st.file_uploader("📤 Upload ảnh giày", type=["jpg", "png", "jpeg"])

if file:
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Ảnh đã upload", use_column_width=True)

    with col2:
        st.info("🔍 Đang phân tích...")

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        idx = np.argmax(pred)
        confidence = float(np.max(pred)) * 100

        st.success(f"👟 Dự đoán: {labels[idx]}")
        st.write(f"📊 Confidence: {confidence:.2f}%")

        st.progress(int(confidence))