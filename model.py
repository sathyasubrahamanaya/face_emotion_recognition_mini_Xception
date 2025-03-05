import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load Haar Cascade face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of emotion labels.
emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

def load_emotion_model(model_path='_mini_xception.100_0.65.hdf5'):
    """
    Loads the saved Keras emotion recognition model.
    """
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def predict_emotion(frame, model):
    """
    Given a BGR image frame and a loaded Keras model, this function:
    - Converts the image to grayscale.
    - Detects faces using Haar cascades.
    - For each face, preprocesses the region of interest and predicts the emotion.
    
    Returns a list of dictionaries with bounding box, predicted label, and probability.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            
            preds = model.predict(roi_gray)[0]
            emotion_probability = np.max(preds)
            label = emotions[preds.argmax()]
            results.append({
                'box': (x, y, w, h),
                'label': label,
                'probability': emotion_probability
            })
    return results
