# Real-time Emotion Recognition App

This repository contains a Streamlit-based application that performs real-time emotion recognition on facial images. The app uses a pre-trained Keras model along with OpenCV for face detection and emotion classification. Users can either upload an image or capture one using their webcam via Streamlit’s built-in `st.camera_input` component.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

The application detects faces in an image and categorizes them into one of seven emotion classes (Angry, Disgust, Scared, Happy, Sad, Surprised, Neutral). The model is built using Keras and optimized for a 48x48 pixel grayscale input. This project is ideal for demonstrations and educational purposes.

## Features

- **Image Upload Mode:** Upload an image file (jpg, jpeg, png) and detect emotions.
- **Webcam Mode:** Capture an image from your webcam using Streamlit’s built-in camera input.
- **Real-time Emotion Detection:** The app uses OpenCV to process images and display detection results with bounding boxes and confidence scores.

## Prerequisites

- Python 3.7 – 3.10 (TensorFlow compatibility)
- pip

> **Note:** The app uses TensorFlow and OpenCV. Make sure you have a compatible Python version as TensorFlow may not support Python 3.11+.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/emotion_recognition_app.git
   cd emotion_recognition_app
   ```

2. **Create and Activate a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Add Your Model:**

   Place your saved Keras model file (e.g., `model.h5`) in the root directory of the project. Ensure the path in `model.py` points to this file.

## Usage

To run the app, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open a new browser window with the Streamlit interface where you can:
- **Upload an Image:** Choose an image file to process and view the emotion detection results.
- **Webcam Mode:** Capture a picture using your webcam and view the detection results.

## Project Structure

```
emotion_recognition_app/
├── app.py            # Main Streamlit application file.
├── model.py          # Contains the Keras model loading and emotion detection functions.
├── requirements.txt  # List of dependencies.
└── model.h5          # Your pre-trained Keras model (ensure this is placed in the root or update the path in model.py).
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

