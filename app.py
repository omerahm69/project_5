import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from app_pages.multipages import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_visual_study_findings import page_visual_study_finding_body
from app_pages.page_hypothesis import page_project_hypothesis_body
from app_pages.page_ML_performance import page_ML_performance_body
from app_pages.page_live_predictions import page_live_predictions_body


app = MultiPage(app_name="")

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Visual Study Findings", page_visual_study_finding_body)
app.add_page("Live Predictions", page_live_predictions_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance", page_ML_performance_body)

app.run()  # Run the app