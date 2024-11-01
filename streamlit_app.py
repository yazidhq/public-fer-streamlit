import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

st.title("Real-Time Facial Emotion Recognition")
st.markdown("![Video Feed](http://127.0.0.1:5000/video_feed)")

# Process video frames in the Flask backend for real-time detection
