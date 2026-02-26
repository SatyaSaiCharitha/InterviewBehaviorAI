# src/real_time_detection.py

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ------------------------------
# 1️⃣ Load Trained Model
# ------------------------------
model = load_model("models/fer_cnn.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ------------------------------
# 2️⃣ Initialize MediaPipe Face Detection
# ------------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ------------------------------
# 3️⃣ Start Webcam
# ------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = x1 + int(bboxC.width * w)
            y2 = y1 + int(bboxC.height * h)

            # Ensure bounding box is within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop face and preprocess
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48,48))
            face_input = np.expand_dims(face_resized, axis=(0,-1)) / 255.0

            # Predict emotion
            prediction = model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(prediction)
            emotion = emotion_labels[emotion_idx]

            # Overlay rectangle + emotion text
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, emotion, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display
    cv2.imshow("Facial Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()