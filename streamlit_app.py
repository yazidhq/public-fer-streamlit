import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import streamlit as st
import time
import os

st.write("Current directory: ", os.getcwd())  # Check current working directory
st.write("Model path exists: ", os.path.exists("Model/Facial_Expression_Recognition.json"))  # Check if model path is valid
