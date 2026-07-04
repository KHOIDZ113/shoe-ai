import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf

# ================= UI =================
st.set_page_config(
    page_title="Shoe AI",
    page_icon="👟",
    layout="centered"
)

st.markdown("<h1 style='text-align:center;'>👟 Shoe AI Classifier</h1>", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("shoe_classifier.h5")

model = load_model()

# ================= LABELS =================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ================= UPLOAD =================
file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred)
    conf = float(np.max(pred)) * 100

    st.success(f"👟 Prediction: {labels[idx]}")
    st.progress(int(conf))
    st.write(f"Confidence: {conf:.2f}%")