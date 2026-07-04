import streamlit as st
import numpy as np
from PIL import Image
import json
import keras

# ================= UI =================
st.set_page_config(page_title="Shoe AI", layout="centered")

st.markdown("<h1 style='text-align:center;'>👟 Shoe AI Classifier</h1>", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return keras.models.load_model("shoe_classifier.h5")

model = load_model()

# ================= LABELS =================
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ================= UPLOAD =================
file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    conf = float(np.max(pred))*100

    st.success(f"Prediction: {labels[idx]}")
    st.progress(int(conf))
    st.write(f"Confidence: {conf:.2f}%")