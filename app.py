import streamlit as st
import numpy as np
import tensorflow as tf
import faiss
from sklearn.preprocessing import normalize
from PIL import Image
from PIL import Image
import numpy as np

st.set_page_config(page_title="Image Similarity Search", layout="wide")

@st.cache_resource
def load_all():
    interpreter = tf.lite.Interpreter(model_path="embedding_model_int8.tflite")
    interpreter.allocate_tensors()

    index = faiss.read_index("faiss.index")
    image_paths = np.load("image_paths.npy", allow_pickle=True)

    return interpreter, index, image_paths

interpreter, index, image_paths = load_all()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess(img):
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return img.astype("float32")

def get_embedding(img):
    img = preprocess(img)[None, ...]
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]["index"])
    return normalize(emb).astype("float32")

st.title("🔍 AI-Powered Image Similarity Search")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Query Image", width=300)

    emb = get_embedding(img_np)
    _, idx = index.search(emb, 5)

    st.subheader("Similar Images")
    cols = st.columns(5)
    for col, i in zip(cols, idx[0]):
        sim_img = Image.open(image_paths[i])
        col.image(sim_img, use_column_width=True)
