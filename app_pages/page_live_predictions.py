import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

def page_live_predictions_body():

    st.title("### Live Predictions")
    st.write ( f"**Visual Study Findings**\n"
        f"*Upload cherry leaf images to predict their health status.)*"
        
)
def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results
    """

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


def resize_input_image(img, version):
    """
    Reshape image to average image size
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image

def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """

    model = load_model(f"outputs/{version}/cherry_leaf_model.h5")
    
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
    model = tf.keras.models.load_model("outputs/v1/my_model.keras")
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
            st.write(f"**Confidence:** {confidence:.2%}")
