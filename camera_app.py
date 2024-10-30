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
st.title("Facial Emotion Recognition")
st.write("Click the button below to start the webcam and detect emotions.")

# Button to start the webcam
if st.button("Start Webcam"):
    # Open a connection to the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not access the webcam. Please check your camera settings.")
    else:
        # Create a placeholder for video feed
        video_placeholder = st.empty()

        # Loop to continuously capture frames from the webcam
        while True:
            ret, img = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break

            # Flip the image to avoid mirroring
            img = cv2.flip(img, 1)

            # Convert to grayscale for face detection
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                roi_gray = gray_img[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                # Predict emotion
                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                predicted_emotion = emotions[max_index]
                cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the image in the Streamlit app
            video_placeholder.image(img, channels="BGR", caption="Real-time Emotion Detection", use_column_width=True)

            # Delay to slow down the loop slightly
            time.sleep(0.1)

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()
