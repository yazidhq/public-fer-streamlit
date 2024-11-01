import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import streamlit as st
import time
from PIL import Image

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

# Use Streamlit's camera input component to capture frames
st.write("Turn on the camera for real-time emotion detection.")
video_input = st.camera_input("Camera")

# Process each frame captured by `st.camera_input`
if video_input is not None:
    # Convert the frame to a format OpenCV can process
    img = Image.open(video_input)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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
    time.sleep(0.1)  # Add a slight delay to control processing speed
