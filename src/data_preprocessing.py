# src/data_preprocessing_folder_v2.py

import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, img_size=(48,48)):
    images = []
    labels = []

    # Emotion mapping: folder name → label
    # src/data_preprocessing_folder_v2.py

    emotion_map = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6
    }
    for emotion_folder, label in emotion_map.items():
        current_folder = os.path.join(folder_path, emotion_folder)
        if not os.path.exists(current_folder):
            continue
        for filename in os.listdir(current_folder):
            img_path = os.path.join(current_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)

    images = np.array(images, dtype='float32')
    images = np.expand_dims(images, -1)
    images /= 255.0
    labels = to_categorical(labels, num_classes=7)

    return images, labels

# -------------------------------
# Function to load full dataset
# -------------------------------
def load_data():
    # Load training images
    train_images, train_labels = load_images_from_folder("data/FER2013/train")
    # Split training into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    # Load test images
    X_test, y_test = load_images_from_folder("data/FER2013/test")

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train, X_val, y_train, y_val, X_test, y_test

# For testing
if __name__ == "__main__":
    X_train, X_val, y_train, y_val, X_test, y_test = load_data()