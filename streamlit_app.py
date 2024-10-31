# Importing the required libraries
import cv2
import streamlit as st

# Function to detect faces in real-time from webcam
def detect_faces():
    # Create the Haar cascade classifier for face detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Create a placeholder for displaying video frames in Streamlit
    video_placeholder = st.empty()

    if not cap.isOpened():
        st.error("Could not access the webcam. Please check your camera settings.")
        return

    # Streamlit session state to control camera
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = True

    # Loop to continuously capture frames from the webcam
    while st.session_state.run_webcam:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is captured correctly
        if not ret:
            st.error("Failed to capture image.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame in Streamlit
        video_placeholder.image(frame, channels="BGR", caption="Real-Time Face Detection")

    # Release the capture and close all resources
    cap.release()

# Streamlit UI
st.title("Real-Time Face Detection")
if st.button("Open Camera"):
    detect_faces()
