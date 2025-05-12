# Pneumonia Detection from Chest X-Ray

A deep learning-based web application to detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). The model is deployed through an interactive Streamlit interface.

---

## Overview

- Developed a binary image classification model to differentiate between normal and pneumonia-affected chest X-rays.
- Achieved **82% accuracy** on the test dataset.
- Designed an interactive web interface for image upload and real-time prediction.
- Aimed to provide a supportive diagnostic tool for faster preliminary detection.

---

## Features

- Upload chest X-ray images in `.jpg`, `.jpeg`, or `.png` formats.
- Real-time prediction displaying whether pneumonia is detected or not.
- Displays the model’s prediction confidence score.
- Simple, clean, and intuitive user interface.

---

## Model Performance

- **Test Accuracy**: 82%
- **Image preprocessing**: Resized to 320x320, rescaled pixel values.
- **CNN model** trained on a publicly available balanced dataset.
- Single-label binary classification: `Normal` or `Pneumonia`.

---

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow (PIL)

Deployed webapp like :- https://pneumoniadetection-lpzrdofaqkuv3n2usqr2rm.streamlit.app/

