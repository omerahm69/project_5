import streamlit as st

import pandas as pd
import numpy as np
import tensorflow as tf
from app_pages.multipages import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary

from app_pages.page_hypothesis import page__hypothesis
from app_pages.page_ML_performance import page_ML_performance

app = MultiPage(app_name="Malaria Detector")

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary)
#app.add_page("Cells Visualiser", page_cells_visualizer_body)
#app.add_page("Malaria Detection", page_malaria_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()  # Run the app