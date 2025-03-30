import os  # Add this import at the top of the file
import cv2
import numpy as np
from tensorflow.keras.models import load_model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.h5')
IMG_SIZE = (256, 256)
MIN_CONFIDENCE = 0.7  # Threshold for considering a valid prediction

class TumorClassifier:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
    
    def is_medical_image(self, image):
        """Basic check if image looks like a medical scan"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        return edge_ratio > 0.01  # Medical images typically have more edges
    
    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, "Invalid image file"
            
            if not self.is_medical_image(img):
                return None, "non_medical"
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            return np.expand_dims(img, axis=[0, -1]), None
        except Exception as e:
            return None, str(e)
    
    def predict(self, image_path):
        preprocessed, error = self.preprocess_image(image_path)
        if error == "non_medical":
            return {
                'prediction': 'Out of Scope',
                'message': 'This does not appear to be a medical scan image'
            }
        elif error:
            return {
                'prediction': 'Error',
                'message': error
            }
        
        prob = self.model.predict(preprocessed)[0][0]
        confidence = max(prob, 1 - prob)
        
        if confidence < MIN_CONFIDENCE:
            return {
                'prediction': 'Uncertain',
                'message': 'The model is not confident about this medical image',
                'probability': float(prob),
                'confidence': float(confidence)
            }
        
        return {
            'prediction': 'Tumor' if prob > 0.5 else 'No Tumor',
            'confidence': float(confidence),
            'probability': float(prob)
        }