import streamlit as st
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import tensorflow as tf
import os


CLASS_NAMES = ["Healthy", "Powdery Mildew"]
IMG_SIZE = (256, 256)

# --- Load model once and cache ---
@st.cache_resource
def load_model():
    model_path = "./jupyter_notebooks/outputs/v1/my_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

# --- Image preprocessing ---
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0) 

# --- Make prediction ---
def predict_image(model, input_tensor):
    prediction = model.predict(input_tensor)
    prob_pm = prediction[0][1]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    return prob_pm, predicted_class

# --- Plot probabilities ---
def plot_predictions_probabilities(prob_pm, predicted_class):
    prob_df = pd.DataFrame({
        "Classify": CLASS_NAMES,
        "Probability": [1 - prob_pm, prob_pm]
    })

    fig = px.bar(
        prob_df,
        x="Classify",
        y="Probability",
        range_y=[0, 1],
        width=600,
        height=300,
        template="seaborn"
    )
    st.plotly_chart(fig)

# --- Main page body ---
def page_live_predictions_body():
    st.title("🍒 Live Leaf Health Prediction")
    st.write("Upload a cherry leaf image (JPG/PNG) to predict if it's **Healthy** or affected by **Powdery Mildew**.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        with st.spinner("Making prediction..."):
            model = load_model()
            if model:
                input_tensor = preprocess_image(image)
                prob_pm, predicted_class = predict_image(model, input_tensor)

                st.write(f"### 🌿 Prediction: **{predicted_class}**")
                st.write(f"**Confidence in 'Powdery Mildew':** {prob_pm:.2%}")
                plot_predictions_probabilities(prob_pm, predicted_class)

