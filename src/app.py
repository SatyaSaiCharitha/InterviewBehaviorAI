# src/main.py

import cv2
from utils import load_emotion_model, init_face_detector, predict_emotion, draw_label

def main():
    model, emotion_labels = load_emotion_model("models/fer_cnn.h5")
    face_detector = init_face_detector(min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = x1 + int(bboxC.width * w)
                y2 = y1 + int(bboxC.height * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                emotion, _ = predict_emotion(face, model, emotion_labels)
                frame = draw_label(frame, (x1,y1,x2,y2), emotion)

        cv2.imshow("Interview Behavior Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()