import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model_path = 'Yolo_Model/best.pt'
model = YOLO(model_path)

def detect_expression(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame

st.title("Real-Time Facial Expression Detection")
st.markdown("Upload a video feed from your webcam and detect facial expressions in real-time!")

camera_active = st.sidebar.checkbox("Activate Camera", value=True)

if camera_active:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible. Please check permissions.")
    else:
        stframe = st.empty()

        while camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = detect_expression(frame_rgb)
            stframe.image(annotated_frame, channels="RGB")

        cap.release()
