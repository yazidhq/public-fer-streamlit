import streamlit as st
import subprocess

# Streamlit app title and instructions
st.title("Facial Emotion Recognition")
st.write("Click the button below to open the camera and detect emotions.")

# Button to open the camera
if st.button("Open Camera"):
    # Run the camera script in a separate process
    subprocess.Popen(["python", "camera_app.py"])
