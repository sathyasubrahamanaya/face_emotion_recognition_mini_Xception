import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import load_emotion_model, predict_emotion

st.title("Real-time Emotion Recognition App")

@st.cache_resource()
def get_model():
    return load_emotion_model()

model = get_model()

# Sidebar: Choose between uploading an image or using the webcam.
app_mode = st.sidebar.selectbox("Choose the App Mode", ["Image Upload", "Webcam"])

if app_mode == "Image Upload":
    st.write("Upload an image to detect emotions:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image file and convert to OpenCV format.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        
        results = predict_emotion(image, model)
        if results:
            for res in results:
                st.write(f"Face at {res['box']}: **{res['label']}** (Confidence: {res['probability']:.2f})")
                (x, y, w, h) = res['box']
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"{res['label']} ({res['probability']:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
        else:
            st.write("No face detected!")
            
elif app_mode == "Webcam":
    st.write("Webcam Mode: Capture an image from your webcam.")
    
    # Use Streamlit's built-in camera_input which uses JavaScript under the hood.
    picture = st.camera_input("Take a picture")
    if picture is not None:
        # Convert the uploaded image (a PIL Image) to an OpenCV BGR image.
        image = np.array(Image.open(picture))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Captured Image", use_column_width=True)
        
        results = predict_emotion(image, model)
        if results:
            for res in results:
                st.write(f"Face at {res['box']}: **{res['label']}** (Confidence: {res['probability']:.2f})")
                (x, y, w, h) = res['box']
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"{res['label']} ({res['probability']:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
        else:
            st.write("No face detected!")
