# src/utils.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# ------------------------------
# 1️⃣ Load CNN Model
# ------------------------------
def load_emotion_model(model_path="models/fer_cnn.h5"):
    model = load_model(model_path)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return model, emotion_labels

# ------------------------------
# 2️⃣ MediaPipe Face Detector
# ------------------------------
def init_face_detector(min_detection_confidence=0.5):
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=min_detection_confidence)
    return face_detector

# ------------------------------
# 3️⃣ Preprocess Face for CNN
# ------------------------------
def preprocess_face(face_image, target_size=(48,48)):
    """
    Input: face_image (BGR)
    Output: normalized grayscale image ready for CNN
    """
    face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, target_size)
    face_input = np.expand_dims(face_resized, axis=(0,-1)) / 255.0
    return face_input

# ------------------------------
# 4️⃣ Predict Emotion from Face
# ------------------------------
def predict_emotion(face_image, model, emotion_labels):
    face_input = preprocess_face(face_image)
    prediction = model.predict(face_input, verbose=0)
    emotion_idx = np.argmax(prediction)
    emotion = emotion_labels[emotion_idx]
    return emotion, prediction

# ------------------------------
# 5️⃣ Draw Bounding Box + Label
# ------------------------------
def draw_label(frame, bbox, label, color=(0,255,0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame