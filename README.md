# Emotion-Detection-Using-Facial-Expression

# 🎭 Emotion Detection Using Facial Expression

Welcome to the **Emotion Detection Using Facial Expression** project! This repository contains a CNN-based model to classify facial expressions into different emotions, as well as a Flask-based web application to demonstrate the model in real-time using a webcam feed. 📸

---

## 📑 Project Overview

This project leverages deep learning to recognize seven different emotions from facial expressions:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

### Key Features
- **Emotion Recognition Model**: Developed using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.
- **Real-Time Detection**: Flask application to capture webcam feed and detect emotions in real-time.
- **Interactive UI**: Simple HTML/CSS interface to display detected emotions.

---

## 📁 Folder Structure

- `FER_2013/` - FER-2013 dataset (can be seen in kaggle also).
- `model/` - Contains the trained model file (`model.h5`).
- `notebooks/` - colab notebook file with model architecture, training, and evaluation.
- `flask_app/` - Flask application with a minimalistic web interface.
    - `app.py` - The main Flask application script.
    - `templates/` - HTML template files.
    - `static/` - Static files (CSS, JS, etc.) for styling the web interface.

---

## 🛠️ Requirements

To run this project, you'll need the following libraries:

- `Flask` - Web framework to serve the application.
- `OpenCV` - For accessing the webcam and processing video frames.
- `Keras` - To load and use the trained CNN model.
- `NumPy` - For numerical operations.
- `Seaborn` & `Matplotlib` - For plotting and visualization in the notebook.
  
Install the dependencies using:

```bash
pip install -r requirements.txt
