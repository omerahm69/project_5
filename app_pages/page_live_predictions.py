import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import tensorflow as tf
import os

# --- Constants ---
CLASS_NAMES = ["Healthy", "Powdery Mildew"]
IMG_SIZE = (256, 256)  # match model input shape

# --- Load model once and cache ---
@st.cache_resource
def load_model():
    model_path = "jupyter_notebooks/outputs/v1/my_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

# --- Image preprocessing ---
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Shape: (1, H, W, 3)

# --- Make prediction ---
def predict_image(model, input_tensor):
    prediction = model.predict(input_tensor)
    prob_pm = prediction[0][1]  # Assuming class 1 is Powdery Mildew
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

"""def page_live_predictions_body():

    st.title("### Live Predictions")
    st.write ( f"**Visual Study Findings**\n"
        f"*Upload cherry leaf images to predict their health status.)*"
        
)
def plot_predictions_probabilities(pred_proba, pred_class):
    
    Plot prediction probability results
    

    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index={'Healthy': 0, 'Powdery_mildew': 1}.keys(),
        columns=['Probability']
    )
    prob_per_class.loc[pred_class] = pred_proba
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Classify'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Classify',
        y=prob_per_class['Probability'],
        range_y=[0, 1],
        width=600, height=300, template='seaborn')
    st.plotly_chart(fig)


#def resize_input_image(img, version):
   # 
    #Reshape image to average image size
#
 #   version ='v1'
  #  with open(f"outputs/{version}/image_shape.pkl", "rb") as f:
   #     image_shape = pickle.load(f)

def resize_input_image(img):
    img = img.convert("RGB")  # Ensure correct channels
    img_resized = img.resize((256, 256), Image.LANCZOS)
    img_array = np.array(img_resized)
    input_tensor = np.expand_dims(img_array, axis=0) / 255.0
    return input_tensor

    #image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    #img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    #img_array = np.array(img_resized)
    #input_tensor = np.expand_dims(img_array, axis=0) / 255.0
    #my_image = np.expand_dims(img_resized, axis=0)/255

    return input_tensor

def load_model_and_predict(my_image, version):
    
    Load and perform ML prediction over live images
    

    model = load_model(f"outputs/{version}/my_model.keras")
    
    pred_proba = model.predict(my_image)[0, 0]

    target_map = {v: k for k, v in {'healthy': 0, 'powdery_mildew': 1}.items()}
    pred_class = target_map[pred_proba > 0.5]
    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis indicates the image is "
        f"**{pred_class.lower()}** powdery_mild.")

    return pred_proba, pred_class

# Load the trained model (load only once)
@st.cache_resource
def load_model():
    import os
    print("Working directory:", os.getcwd())
    print("Model file exists:", os.path.exists("C:\\python_projects\\project_5\\jupyter_notebooks\\outputs\\v1\\my_model.keras"))

    model = tf.keras.models.load_model("jupyter_notebooks/outputs/v1/my_model.keras")
    return model

# Preprocessing function
def preprocess_image(img):
    img = img.convert("RGB")  # Convert RGBA to RGB
    img = img.resize((128, 128))  # Match model's input size
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

def page_live_predictions_body():
    st.write("### 🍒 Live Leaf Health Prediction")
    st.info("Upload a cherry leaf image (JPG/PNG) and the model will predict if it's **Healthy** or **Powdery Mildew** affected.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        with st.spinner("Making prediction..."):
            model = load_model()
            input_tensor = preprocess_image(image)
            model=load_model()
            prediction = model.predict(input_tensor)

            class_names = ["Healthy", "Powdery Mildew"]
            predicted_class = class_names[int(np.round(prediction[0][0]))]
            confidence = prediction[0][0] if predicted_class == "Powdery Mildew" else 1 - prediction[0][0]

            st.write(f"### 🌿 Prediction: **{predicted_class}**")
            st.write(f"**Confidence:** {confidence:.2%}")"""
