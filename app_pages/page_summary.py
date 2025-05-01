import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():
    
    st.write("###  Quick Project Summary")
    
st.info(
    f"""**General Information**
* The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

* To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

**Goal**: Train a machine learning model to classify images into these two categories:
    1. **Healthy**
    2. **Powdery Mildew**

**Project Dataset**
* The dataset used for this project is from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.
"""
)
st.info(
        f"**General Information**\n"
        f"* The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.*"

        f"To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops. \n"
        f"""**Goal**: Train a machine learning model to classify images into these two categories.
            1. **Healthy**
            2. **Powdery Mildew**\n"

        **Project Dataset**
        * The dataset used for this project is from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
        The dataset contains +4 thousand images taken from the client's
        crop fields.
        The images show healthy cherry leaves and cherry leaves
        that have powdery mildew, a fungal disease that affects many plant species.
        The cherry plantation crop is one of the finest products in their portfolio,
        and the company is concerned about supplying the market with a compromised quality product. "
        
        "The images show healthy cherry leaves and cherry leaves that
        have powdery mildew.""")

st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/codeinstitute/cherry-leaves/main/README.md).")
    
st.success(
        
    f"""The project has two business requirements:
* 1 - The client is interested in conducting a study to visually differentiate
    healthy cherry leaves from those with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf
    is healthy or contains powdery mildew."""
)