
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os

st.set_page_config(page_title="Shoe AI Pro", page_icon="👟", layout="centered")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#0f172a,#1e293b);color:white}
.card{background:rgba(30,41,59,0.8);padding:20px;border-radius:20px}
.title{text-align:center;font-size:40px;font-weight:800;color:#38bdf8}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>👟 Shoe AI Pro Final</div>", unsafe_allow_html=True)

model=tf.keras.models.load_model("shoe_classifier.h5")

# load labels if exists
if os.path.exists("class_indices.json"):
    classes=json.load(open("class_indices.json"))
    idx_to_class={v:k for k,v in classes.items()}
else:
    n=model.output_shape[-1]
    idx_to_class={i:f"class_{i}" for i in range(n)}

def predict(img):
    img=img.resize((224,224))
    x=np.array(img)/255.0
    x=np.expand_dims(x,0)
    p=model.predict(x,verbose=0)[0]
    top3=np.argsort(p)[::-1][:3]
    return [(idx_to_class[i],float(p[i])) for i in top3]

up=st.file_uploader("Upload image",type=["png","jpg","jpeg"])

if up:
    img=Image.open(up)
    st.image(img,use_container_width=True)

    if st.button("Predict 🚀"):
        res=predict(img)
        st.markdown("<div class='card'>",unsafe_allow_html=True)
        st.write("### Top Result:",res[0][0])
        for k,v in res:
            st.write(k,":",round(v,3))
        st.markdown("</div>",unsafe_allow_html=True)
