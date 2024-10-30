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
st.write("Use your webcam to detect emotions in real-time.")

# Start video capture from the webcam
video_capture = cv2.VideoCapture(0)

# Create a placeholder for the video stream
frame_placeholder = st.empty()

# Streamlit app loop
while True:
    # Capture frame-by-frame
    ret, img = video_capture.read()
    if not ret:
        st.warning("Unable to access the webcam.")
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

    # Update the image in the Streamlit app
    frame_placeholder.image(img, channels="BGR", caption="Real-time Emotion Detection", use_column_width=True)

    # Stop button to break the loop
    if st.button('Stop Webcam'):
        break

# Release the video capture when done
video_capture.release()
cv2.destroyAllWindows()
