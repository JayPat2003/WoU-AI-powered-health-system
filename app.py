from flask import Flask, render_template, request, send_from_directory, redirect
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import torch
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd 
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'supersecretkey'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

@app.route('/')
def index():
    return render_template('index.html')

## KIDNEY STONE DETECTION ##
kidney_stone_model = YOLO('kidney_stone_model.pt')

@app.route('/kidney_stone', methods=['GET', 'POST'])
def kidney_stone():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read the image using OpenCV
            img = cv2.imread(filepath)

            # Perform detection using YOLOv8 model
            results = kidney_stone_model(img)

            if results:
                # List to keep track of label positions to prevent overlap
                label_positions = []

                # Process each result
                for result in results:
                    # Access detection boxes
                    boxes = result.boxes
                    if boxes is not None:
                        # Iterate over each detection
                        for box in boxes.xyxy:
                            try:
                                # Unpack bounding box coordinates
                                x1, y1, x2, y2 = map(float, box[:4])
                                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                                # Draw bounding boxes on the image
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # Prepare text for label
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.2  # Small font size
                                font_color = (0, 255, 255)  # Yellow text color
                                thickness = 1
                                label = "kidney stone"

                                # Label position (by default, place above the box)
                                label_position = (x1, y1 - 10)

                                # Check if this label overlaps with previous ones
                                def is_overlapping(new_label_pos, existing_label_pos):
                                    new_x, new_y = new_label_pos
                                    existing_x, existing_y = existing_label_pos
                                    distance_threshold = 20  # Minimum pixel distance between labels
                                    return abs(new_x - existing_x) < distance_threshold and abs(new_y - existing_y) < distance_threshold

                                # Adjust the position to avoid overlap with previous labels
                                while any(is_overlapping(label_position, pos) for pos in label_positions):
                                    label_position = (label_position[0], label_position[1] - 15)  # Move the label upward by 15 pixels

                                # Store the new label position to track it
                                label_positions.append(label_position)

                                # Add label to the image
                                cv2.putText(img, label, label_position, font, font_scale, font_color, thickness, cv2.LINE_AA)

                            except IndexError as e:
                                print("IndexError:", e)
                            except ValueError as e:
                                print("ValueError:", e)

                # Save the result image
                output_filename = 'output_' + filename
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                cv2.imwrite(output_path, img)

                # Render template with the processed image
                return render_template('kidney_stone.html', image_url=output_filename)
            else:
                return render_template('kidney_stone.html', error="No detections found.")
    return render_template('kidney_stone.html')



## CANCER DETECTION ##
cancer_detection_model = load_model('resnet50_histopathology_model.h5')

@app.route('/cancer_detection', methods=['GET', 'POST'])
def cancer_detection():
    if request.method == 'POST':
        # Get the uploaded file from the form
        file = request.files['image']
        if file and allowed_file(file.filename):
            # Secure the filename and save it
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # If the file is in TIFF format, convert to PNG or JPG
            if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
                img = Image.open(filepath)
                converted_filename = filename.rsplit('.', 1)[0] + '.png'  # Change the extension to PNG
                converted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)
                img.convert('RGB').save(converted_filepath, 'PNG')  # Convert to PNG
                filepath = converted_filepath  # Update the path to the converted image
                filename = converted_filename

            # Predict the result and confidence score
            result = predict_cancer_image(filepath, (224, 224))
            
            # Render the result on the HTML template
            return render_template('cancer_detection.html', result=result, image_url=filename)
    
    return render_template('cancer_detection.html')

# Preprocess image (same as during training)
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")  # Convert to RGB if it's in grayscale
    image = image.resize(target_size)  # Resize the image
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


def predict_cancer_image(image_path, target_size):
    # Preprocess the image for prediction
    image = preprocess_image(image_path, target_size)
    
    # Make the prediction
    prediction = cancer_detection_model.predict(image)
    
    # Evaluate the prediction result
    if prediction >= 0.5:
        result = "Cancer Detected"
    else:
        result = "Cancer Not Detected"

    return result


## LIVER FIBROSIS ##
try:
    liver_fibrosis_model = load_model('liver_fibrosis_xception_model.h5')
    print("Liver fibrosis model loaded successfully.")
except Exception as e:
    print(f"Error loading liver fibrosis model: {e}")

    
class_label = ['f0', 'f1f2f3', 'f4']

@app.route('/liver_disease', methods=['GET', 'POST'])
def liver_disease():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to {filepath}")
            prediction_result = predict_liver_fibrosis(filepath)
            
            return render_template('liver_disease.html', filename=filename, result=prediction_result)
    
    return render_template('liver_disease.html')


def predict_liver_fibrosis(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = liver_fibrosis_model.predict(img_array)
    print(f"Prediction: {prediction}")
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_label[predicted_class_index]
    print(f"Predicted label: {predicted_label}")
    
    if predicted_label == 'f0':
        return "No liver fibrosis detected."
    elif predicted_label == 'f1f2f3':
        return "Liver fibrosis detected at early stage."
    elif predicted_label == 'f4':
        return "Liver fibrosis detected in stage f4."
    else:
        return "Error in prediction."




## DIABETIC RETINOPATHY ##
class_labels = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
model = load_model('efficientnet_dr_model.h5')

# Function to predict only the class
def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).resize((224, 224))  
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict using the model
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0] 
    return class_labels[predicted_class] 

@app.route('/diabetic_retinopathy', methods=['GET', 'POST'])
def diabetic_retinopathy():
    result = None
    image_url = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict
            predicted_class = predict_image(file_path)
            result = f"Prediction: {predicted_class}"
            image_url = filename  # To display the image

    return render_template('diabetic_retinopathy.html', result=result, image_url=image_url)

import logging
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
    
    
