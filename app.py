import os
os.environ["KERAS_BACKEND"] = "jax"

import streamlit as st
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    import keras
    return keras.models.load_model("mobile_net_skin_model.keras")

# Must match EXACTLY the folder names in your dataset (alphabetical order)
class_names = [
    "Acne",
    "Actinic_Keratosis",
    "Benign_tumors",
    "Bullous",
    "Candidiasis",
    "DrugEruption",
    "Eczema",
    "Infestations_Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrh_Keratoses",
    "SkinCancer",
    "Sun_Sunlight_Damage",
    "Tinea",
    "Unknown_Normal",
    "Vascular_Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts"
]

st.title("Skin Disease Classification App")
st.write("Upload an image and the model will predict the skin disease class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model()
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")
