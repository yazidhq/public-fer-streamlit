import cv2
import streamlit as st

# Streamlit app
st.title("Webcam Test")

# Placeholder for video feed
video_placeholder = st.empty()

# Open a connection to the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Could not access the webcam. Please check your camera settings.")
else:
    ret, frame = cap.read()
    if ret:
        video_placeholder.image(frame, channels="BGR", caption="Webcam Feed", use_column_width=True)
    else:
        st.error("Failed to capture image.")

cap.release()
