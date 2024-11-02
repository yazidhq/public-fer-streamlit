import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import streamlit as st
import threading

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
st.write("Click the button below to open the camera and detect emotions.")

# Function to run the webcam and detect emotions
def run_camera():
    # Open a connection to the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access the webcam. Please check your camera settings.")
        return
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
        # Display the image in an OpenCV window
        cv2.imshow('Real-time Emotion Detection', img)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
# Button to open the camera
if st.button("Open Camera"):
    # Run the camera in a separate thread
    threading.Thread(target=run_camera).start()

# import streamlit as st

# st.title("Facial Emotion Recognition")

# # Embed the Flask video stream in Streamlit
# st.markdown("""<iframe src="http://localhost:5000" frameborder="0" allowfullscreen></iframe>""",unsafe_allow_html=True)
