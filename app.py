import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cache model so it only loads once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobile_net_skin_model.keras")

class_names = [
    "Acne",
    "Actinic Keratosis",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Eczema",
    "Impetigo",
    "Lichen Planus",
    "Melanoma",
    "Molluscum Contagiosum",
    "Psoriasis",
    "Ringworm",
    "Rosacea",
    "Scabies",
    "Seborrheic Keratosis",
    "Skin Cancer",
    "Squamous Cell Carcinoma",
    "Tinea Versicolor",
    "Urticaria (Hives)",
    "Vitiligo",
    "Warts"
]

st.title("Skin Disease Classification App")
st.write("Upload an image and the model will predict the skin disease class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  # fixed

    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    model = load_model()  # cached
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")  # e.g. 87.43% instead of 0.87
