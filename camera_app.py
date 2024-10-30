import streamlit as st
import subprocess
import os

# Streamlit app title and instructions
st.title("Facial Emotion Recognition")
st.write("Click the button below to open the camera and detect emotions.")

# Button to open the camera
if st.button("Open Camera"):
    # Ensure the camera_app.py script is accessible
    if os.path.isfile("camera_app.py"):
        # Run the camera script in a separate process
        subprocess.Popen(["python", "camera_app.py"])
        st.success("Camera is opening...")
    else:
        st.error("camera_app.py not found!")
