import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import streamlit as st
import time

# Load model architecture from JSON
with open("Model/Facial_Expression_Recognition.json", "r") as json_file:
    model = model_from_json(json_file.read())
# Load weights from the correct path
model.load_weights("Model/fer.weights.h5")
# Load Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier('Model/haarcascade_frontalface_default.xml')
# Define emotions
emotions = ('ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL')

# Streamlit app
st.title("Real-Time Facial Emotion Recognition")

# Placeholder for video feed
video_placeholder = st.empty()

cap = st.camera_input("Camera")
if cap is not None:
    st.error("Could not access the webcam. Please check your camera settings.")
else:
    # Manage webcam start/stop functionality with buttons
    start_button = st.button("Start Webcam", key="start_webcam")
    stop_button = st.button("Stop Webcam", key="stop_webcam")
    
    # Webcam session state
    if start_button:
        st.session_state.run_webcam = True
    if stop_button:
        st.session_state.run_webcam = False

    # Main loop to capture frames from the webcam
    while st.session_state.get("run_webcam", False):
        ret, img = cap.read()
        
        # Check if the frame was captured successfully
        if not ret:
            st.error("Failed to capture image.")
            break

        # Flip the image to avoid mirroring
        img = cv2.flip(img, 1)

        # Convert to grayscale for face detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)
        
        # Detect and annotate emotions on faces
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=1)
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            # Predict emotion
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]

            # Annotate the image with the predicted emotion
            cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Display the processed frame in the Streamlit app
        video_placeholder.image(img, channels="BGR", caption="Real-time Emotion Detection", use_column_width=True)

        # Short delay to control frame rate
        time.sleep(0.1)

    # Release the webcam after the loop exits
    cap.release()
