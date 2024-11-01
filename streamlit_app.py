import numpy as np
import cv2
from keras.models import model_from_json
from keras.preprocessing import image
import streamlit as st

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
st.title("Facial Emotion Recognition")

# Start video capture from the webcam using Streamlit's camera component
video_input = st.camera_input("", key="webcam")

if video_input is not None:
    # Convert the image to an array
    img = cv2.imdecode(np.frombuffer(video_input.read(), np.uint8), cv2.IMREAD_COLOR)

    # Flip the image to avoid mirroring
    img = cv2.flip(img, 1)

    # Convert to grayscale for face detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

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
        cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    # Display the image in the Streamlit app
    st.image(img, channels="BGR", caption="Result", use_column_width=True)