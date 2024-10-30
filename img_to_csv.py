from PIL import Image
import pandas as pd
import numpy as np
import os

# Define directory paths
dataset_dir = r'Data Image\test'
output_csv = 'Data Image/test.csv'

# Create a dictionary to store data
data = {'emotion': [], 'pixels': [], 'Usage': []}

# Map labels to emotions based on your folder structure
emotion_labels = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

# Iterate through each emotion folder
for emotion, label in emotion_labels.items():
    emotion_path = os.path.join(dataset_dir, emotion)
    if not os.path.isdir(emotion_path):
        continue
    
    # Process each image in the emotion folder
    for img_file in os.listdir(emotion_path):
        img_path = os.path.join(emotion_path, img_file)
        try:
            # Open and convert image to grayscale, resize if needed
            img = Image.open(img_path).convert('L')  # 'L' mode for grayscale
            img = img.resize((48, 48))  # Resize to 48x48 pixels for consistency
            
            # Convert image to pixels array
            pixels = np.array(img).flatten()  # Flatten 2D array to 1D
            pixels_str = ' '.join(map(str, pixels))  # Convert to space-separated string
            
            # Add to data dictionary
            data['emotion'].append(label)
            data['pixels'].append(pixels_str)
            data['Usage'].append('Testing')  # Or 'Test' if applicable
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

# Convert dictionary to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"Dataset saved to {output_csv}")
