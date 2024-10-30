
import streamlit as st
import os

st.write("Current directory: ", os.getcwd())  # Check current working directory
st.write("Model path exists: ", os.path.exists("Model/Facial_Expression_Recognition.json"))  # Check if model path is valid
