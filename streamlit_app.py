import cv2
import streamlit as st

st.title("Webcam Test")

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Could not access the webcam. Please check your camera settings.")
else:
    ret, frame = cap.read()
    if ret:
        st.image(frame, channels="BGR", caption="Webcam Feed", use_column_width=True)
    else:
        st.error("Failed to capture image.")

cap.release()
