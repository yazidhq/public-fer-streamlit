import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model_path = 'Yolo_Model/stranger-model.pt'
model = YOLO(model_path)

def detect_expression(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame

st.title("Facial Emotion Detection")
st.markdown("Upload a photo or use your webcam to detect facial expressions!")

camera_active = st.sidebar.checkbox("Activate Camera", value=False)

if camera_active:
    st.sidebar.info("Photo upload is disabled while the camera is active.")
    upload_photo = None
else:
    upload_photo = st.sidebar.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

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

if upload_photo:
    file_bytes = np.asarray(bytearray(upload_photo.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_column_width=True, channels="RGB")
    
    annotated_image = detect_expression(image_rgb)
    st.image(annotated_image, caption="Detected Expressions", use_column_width=True, channels="RGB")
