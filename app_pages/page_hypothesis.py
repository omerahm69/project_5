import streamlit as st
import matplotlib.pyplot as plt

def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success("""
    * **Hypothesis:** A deep learning model, particularly a Convolutional Neural Network (CNN), can effectively classify cherry leaf images into 'Healthy' and 'Powdery Mildew' categories with high accuracy. 

    * **Rationale:** Healthy leaves are typically uniform in green color and free from white spots, while leaves affected by powdery mildew exhibit noticeable white or grayish fungal growth patterns on their surface.

    * **Validation Approach:**
        - A balanced dataset was used with sufficient examples of both classes.
        - Transfer learning techniques were applied to improve model performance.

    * **Observations:**
        - Image montages suggest distinguishable visual patterns between healthy and infected leaves.
        - However, statistical analyses such as average image, variability image, and difference between class averages did not reveal a distinct pattern.
    """)