import streamlit as st
import matplotlib.pyplot as plt


def page_visual_study_finding_body():

    st.write("### Visual Study Findings")

    st.info ( f"""**Visual Study Findings**\n
        * After analyzing the dataset, here are the findings to visually differentiate a healthy cherry leaf from one with powdery mildew:
            - Healthy leaves are uniform in color, typically green, and free from white spots.
            - Leaves affected by powdery mildew show noticeable white or grayish fungal growth patterns on their surface. * """
        
)
    
    st.write("---")
    st.write("### Sample Images")

    col1, col2 = st.columns(2)
    with col1:
        st.image("outputs/v1/avg_var_healthy.png", caption="Healthy Cherry Leaf", use_container_width=True)
    with col2:
        st.image("outputs/v1/avg_var_powdery_mildew.png", caption="Leaf with Powdery Mildew", use_container_width=True)

    st.write("---")
    st.write("### Image Analysis Techniques")

    st.markdown("""
    - **Image Montage**: Created to visually scan large numbers of leaves side-by-side.
    - **Average Image**: Shows the typical structure of each class (healthy vs Powdery Mildew).
    - **Image Variability**: Highlights areas of inconsistency (noise, disease presence).
    - **Difference of Averages**: Emphasized faint patterns of mildew that are hard to spot in raw images.
    """)

   # st.image("outputs/v1/image_montage.png", caption="Image Montage of Samples", use_container_width=True)
    st.image("outputs/v1/avg_diff.png", caption="Difference Between Average Healthy and Powdery Mildew Leaves", use_container_width=True)