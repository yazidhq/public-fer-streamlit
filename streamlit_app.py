# Importing the required libraries
import cv2
import streamlit as st
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import time

# Load model architecture and weights once at the start
@st.cache_resource
def load_emotion_model():
    with open("Model/Facial_Expression_Recognition.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("Model/fer.weights.h5")
    return model

# Load Haar Cascade for face detection once at the start
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier('Model/haarcascade_frontalface_default.xml')

# Function to detect faces in real-time from webcam
def detect_faces():
    model = load_emotion_model()
    face_haar_cascade = load_face_cascade()
    emotions = ('ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL')

    st.title("Facial Emotion Recognition")
    st.write("Use your webcam to detect emotions in real-time.")
    video_placeholder = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not access the webcam. Please check your camera settings.")
        return

    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False

    start_button = st.button("Start Webcam", key="start_webcam")
    stop_button = st.button("Stop Webcam", key="stop_webcam")
    
    if start_button:
        st.session_state.run_webcam = True
    if stop_button:
        st.session_state.run_webcam = False

    while st.session_state.run_webcam:
        ret, img = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        img = cv2.flip(img, 1)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)
        
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0
            
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]
            cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        video_placeholder.image(img, channels="BGR", caption="Real-time Emotion Detection", use_column_width=True)
        time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI
st.title("Real-Time Face Detection")

if st.button("Open Camera"):
    detect_faces()
