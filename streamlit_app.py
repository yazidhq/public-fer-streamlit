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

# Load DNN model for face detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

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

    # Prepare image for DNN
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    h, w = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Threshold for detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), thickness=1)

            roi_gray = cv2.cvtColor(img[y:y1, x:x1], cv2.COLOR_BGR2GRAY)
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
