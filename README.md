#  Interview Behavior AI

##  Overview
An AI-powered system that analyzes facial expressions during interviews to detect emotions and behaviors such as confidence, attentiveness, and stress.  
The system uses computer vision and machine learning to provide actionable insights for interview preparation and candidate evaluation.

---

##  Tech Stack
- Python
- OpenCV
- Dlib / Mediapipe (for facial landmarks)
- TensorFlow / PyTorch (for emotion detection model)
- Scikit-learn
- NumPy
- Pandas
- Streamlit / Flask (for app deployment)

---

##  Features
- Real-time facial expression detection
- Emotion classification (e.g., happy, neutral, sad, surprised, etc.)
- Confidence and attentiveness analysis
- Visualization of detected emotions
- Optional logging of results for evaluation

---

##  Project Structure
InterviewBehaviorAI/
│
├── data/             — Dataset folder (FER2013 images)  
├── models/           — Trained models (e.g., `fer_cnn.h5`)  
├── src/              — Source code for preprocessing, training, and real-time detection  
│   ├── app.py  
│   ├── train_model.py  
│   ├── real_time_detection.py  
│   ├── data_preprocessing.py  
│   └── utils.py  
├── main.py           — Main application entry point  
├── README.md  
└── .gitignore  

---

##  How to Run

1. Install dependencies:
pip install -r requirements.txt
2. Run the application:
python main.py
3. (Optional) Explore notebooks or scripts for training and evaluation:
jupyter notebook

# Future Improvements

Improve model accuracy with larger datasets

Real-time dashboard for emotion tracking

Cloud deployment for web access

Integration with interview scoring metrics

# Author
Sai Charitha
Aspiring Machine Learning Engineer

