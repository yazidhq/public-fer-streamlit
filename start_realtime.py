import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Load model architecture from JSON
with open("Model/Facial_Expression_Recognition.json", "r") as json_file:
    model = model_from_json(json_file.read())

# Load weights from the correct path
model.load_weights("Model/fer.weights.h5")

# Load Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier('Model/haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define emotions
emotions = ('ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL')

# Create a named window and set its size
cv2.namedWindow("Facial Emotion Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Facial Emotion Analysis", 1200, 900)  # Set the desired window size

while True:
    ret, test_img = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip the image to avoid mirroring
    test_img = cv2.flip(test_img, 1)

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=1)
        
        # Region of Interest (ROI) for the detected face
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to 48x48 for model input
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # Normalize pixel values

        # Predict emotion
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # Display the predicted emotion
        cv2.putText(test_img, predicted_emotion, (int(x), int(y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Draw the bar chart
        bar_width = 10  # Set the bar width smaller
        spacing = 15    # Increase the spacing between bars
        for i in range(len(emotions)):
            # Height of each bar based on prediction
            bar_height = int(predictions[0][i] * 200)  # Scale to half the height of the previous chart for smaller size
            # Draw the bar for each emotion
            cv2.rectangle(test_img, 
                        (20 + (bar_width + spacing) * i, 350),  # Top-left corner (moved up)
                        (20 + (bar_width + spacing) * i + bar_width, 350 - bar_height),  # Bottom-right corner (moved up)
                        (0, 255, 0),  # Color for the bars
                        -1)  # Fill the rectangle
            
            # Display the emotion name vertically (portrait mode) with added margin
            label_x = 20 + (bar_width + spacing) * i  # X position for the label
            label_y = 370  # Adjust starting Y position for the label (increased margin)

            for j in range(len(emotions[i])):  # Loop through each character in the emotion
                cv2.putText(test_img, emotions[i][j], 
                            (label_x, label_y + j * 15),  # Adjust Y position for each character
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 255, 0), 1)  # Color and thickness of the text

    # Display the resulting image with rectangles and predictions
    cv2.imshow("Facial Emotion Analysis", test_img)  # Use the named window

    # Exit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
