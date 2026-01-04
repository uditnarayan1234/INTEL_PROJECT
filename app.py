import streamlit as st
import numpy as np
import cv2
import faiss
import tensorflow as tf
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Image Similarity Search", layout="wide")

# ---------------------------
# LOAD MODEL & DATA
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("embedding_model.h5", compile=False)

@st.cache_resource
def load_faiss():
    return faiss.read_index("faiss.index")

@st.cache_data
def load_paths():
    return np.load("image_paths.npy", allow_pickle=True)

model = load_model()
index = load_faiss()
image_paths = load_paths()

# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img.astype("float32")

# ---------------------------
# SEARCH FUNCTION
# ---------------------------
def search_similar(img, k=5):
    img = preprocess_image(img)
    emb = model.predict(img[np.newaxis, ...], verbose=0)
    emb = normalize(emb).astype("float32")

    distances, indices = index.search(emb, k)
    return [image_paths[i] for i in indices[0]]

# ---------------------------
# UI
# ---------------------------
st.title("🧠 AI-Powered Image Similarity Search")
st.markdown("Upload an image to find visually similar fashion products.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Query Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300)

    if st.button("Find Similar Images"):
        with st.spinner("Searching..."):
            results = search_similar(img, k=5)

        st.subheader("Similar Images")
        cols = st.columns(5)

        for col, path in zip(cols, results):
            simg = cv2.imread(path)
            simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
            col.image(simg, use_column_width=True)
