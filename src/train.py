import tensorflow as tf  # <-- Add this import
import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Rest of your existing code...
# Path configurations
TRAIN_TUMOR_PATH = 'data/train/tumor/'
TRAIN_NO_TUMOR_PATH = 'data/train/no tumor/'
MODEL_SAVE_PATH = 'models/best_model.h5'
IMG_SIZE = (256, 256)

def load_data(tumor_path, no_tumor_path):
    images = []
    labels = []
    
    # Load tumor images
    for img_name in os.listdir(tumor_path):
        img = cv2.imread(os.path.join(tumor_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        images.append(img)
        labels.append(1)  # 1 for tumor
        
    # Load no tumor images
    for img_name in os.listdir(no_tumor_path):
        img = cv2.imread(os.path.join(no_tumor_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        images.append(img)
        labels.append(0)  # 0 for no tumor
    
    return np.array(images), np.array(labels)

def build_model(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Classifier
    flatten = tf.keras.layers.Flatten()(pool2)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)
    
    return Model(inputs=inputs, outputs=output)

# Load and prepare data
X, y = load_data(TRAIN_TUMOR_PATH, TRAIN_NO_TUMOR_PATH)
X = np.expand_dims(X, axis=-1)  # Add channel dimension
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile model
model = build_model()
model.compile(optimizer=Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train with F1-score monitoring
class F1Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_pred = (model.predict(X_val) > 0.5).astype(int)
        f1 = f1_score(y_val, val_pred)
        print(f" - val_f1: {f1:.4f}")
        logs['val_f1'] = f1

history = model.fit(X_train, y_train,
                   validation_data=(X_val, y_val),
                   epochs=30,
                   batch_size=32,
                   callbacks=[F1Callback()])

# Save model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")