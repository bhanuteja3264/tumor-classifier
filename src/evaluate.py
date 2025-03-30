import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, f1_score

# Path configurations
TEST_TUMOR_PATH = 'data/test/tumor/'
TEST_NO_TUMOR_PATH = 'data/test/no tumor/'
MODEL_PATH = 'models/best_model.h5'
IMG_SIZE = (256, 256)

def load_test_data():
    images = []
    labels = []
    
    # Load tumor images
    for img_name in os.listdir(TEST_TUMOR_PATH):
        img = cv2.imread(os.path.join(TEST_TUMOR_PATH, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        images.append(img)
        labels.append(1)
        
    # Load no tumor images
    for img_name in os.listdir(TEST_NO_TUMOR_PATH):
        img = cv2.imread(os.path.join(TEST_NO_TUMOR_PATH, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        images.append(img)
        labels.append(0)
    
    return np.array(images), np.array(labels)

# Load model and test data
model = load_model(MODEL_PATH)
X_test, y_test = load_test_data()
X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
f1 = f1_score(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))
print(f"\nF1-Score: {f1:.4f}")

# Save predictions
np.save('models/test_predictions.npy', y_pred)
np.save('models/test_labels.npy', y_test)