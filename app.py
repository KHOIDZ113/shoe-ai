import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Shoe AI", layout="centered")

st.title("👟 Shoe Classifier AI")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("shoe_classifier.h5")

model = load_model()

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    pred = model.predict(img)
    idx = np.argmax(pred)

    st.success(labels[idx])