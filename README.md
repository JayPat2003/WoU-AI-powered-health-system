# WoU-AI-Powered-Health-System

This repository contains a Flask-based web application that integrates four deep learning models for detecting various medical conditions from images. The models include:
- **Kidney Stone Detection**
- **Diabetic Retinopathy Detection**
- **Liver Fibrosis Histopathology Detection**
- **Histopathology Metastasis Lymph Node Cancer Detection**


## Overview

This project provides a unified interface for medical professionals and researchers to use deep learning models for diagnosing different conditions from medical images. The interface allows users to upload images, select a model, and obtain predictions from the model along with visualized results.

### Models Included
1. **Kidney Stone Detection**: Identifies kidney stones from ultrasound or CT scan images, providing bounding boxes around detected stones.
2. **Diabetic Retinopathy Detection**: Classifies retinal images into various stages of diabetic retinopathy to aid in early diagnosis and treatment.
3. **Liver Fibrosis Histopathology Detection**: Analyzes liver tissue histopathology slides to detect signs and levels of fibrosis, aiding in liver disease diagnosis.
4. **Histopathology Metastasis Lymph Node Cancer Detection**: Detects the presence of cancerous cells in lymph node histopathology slides, providing insights into the spread of cancer.

## Prerequisites

Before running the application, ensure that you have the following:
- **Python 3.8+**
- **Virtual Environment (recommended)** for managing dependencies

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt


## Usage (explanation with relevant screenshots)

1. **Navigate to the Web Interface**: Run the application and go to `http://127.0.0.1:5000/`.
2. **Select a Model**: Choose the desired model from the interface.
3. **Upload an Image**: Upload an image in a supported format (e.g., `.jpg`, `.png`).
4. **Get Prediction**: Click the "Submit" button to obtain predictions from the model. The result will include the processed image with annotations (if applicable) and the predicted class or probability.

## Model Details

- **Kidney Stone Detection**: 
  - **Model Type**: Object Detection using YOLOv8.
  - **Dataset**: Trained on a dataset of kidney stone images. 
  - **Accuracy**: 
  - **Purpose**: Identifies and localizes kidney stones in medical imaging, providing bounding boxes around detected stones.
  - **Additional Information**: 

- **Diabetic Retinopathy Detection**: 
  - **Model Type**: Classification using a Convolutional Neural Network (CNN).
  - **Dataset**: Trained on a dataset of retinal images with various stages of diabetic retinopathy.
  - **Accuracy**: 
  - **Purpose**: Classifies retinal images into different stages of diabetic retinopathy, aiding early detection and management.
  - **Additional Information**: 

- **Liver Fibrosis Histopathology Detection**: 
  - **Model Type**: Deep Learning-based Classification.
  - **Dataset**: Trained on histopathological liver tissue slides.
  - **Accuracy**: 
  - **Purpose**: Analyzes liver tissue slides to classify the level of fibrosis, supporting liver disease diagnosis.
  - **Additional Information**: 

- **Histopathology Metastasis Lymph Node Cancer Detection**: 
  - **Model Type**: Deep Learning-based Detection.
  - **Dataset**: Trained on histopathological slides of lymph node tissues.
  - **Accuracy**: 
  - **Purpose**: Detects metastasis in lymph node tissue slides, assisting in the identification of cancer spread.
  - **Additional Information**: 


## Contributing

If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.




