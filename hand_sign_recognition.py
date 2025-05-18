#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time


# In[2]:


# Configure TensorFlow to be less verbose and use memory more efficiently
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging verbosity
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


# In[3]:


# Constants
MODEL_PATH = 'hand_sign_model.h5'
IMAGE_SIZE = (64, 64)


# In[4]:


# Load the class labels
def load_labels(dataset_path):
    try:
        return sorted(os.listdir(dataset_path))
    except Exception as e:
        print(f"Error loading labels: {e}")
        # Fallback to alphabet if dataset path is invalid
        return [chr(i) for i in range(ord('A'), ord('Z')+1)]


# In[5]:


# Load or create model
def get_model(dataset_path=None, force_train=False):
    """
    Load existing model or train a new one if needed
    
    Args:
        dataset_path: Path to training data (required only for training)
        force_train: Whether to force training even if model exists
        
    Returns:
        model: Loaded or trained TensorFlow model
        labels: List of class labels
    """
    # Try to load labels first
    if dataset_path:
        labels = load_labels(dataset_path)
    else:
        # If no dataset path and no existing model, we can't proceed
        if not os.path.exists(MODEL_PATH) and not force_train:
            raise ValueError("No model exists and no dataset provided for training")
        # Try to infer labels from alphabet (A-Z)
        labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    # Check if model already exists and we're not forcing a retrain
    if os.path.exists(MODEL_PATH) and not force_train:
        print(f"Loading existing model from {MODEL_PATH}...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
            return model, labels
        except Exception as e:
            print(f"Error loading model: {e}")
            if not dataset_path:
                raise ValueError("Failed to load model and no dataset provided for training")
            print("Will train a new model instead.")
    
    # If we get here, we need to train a new model
    if not dataset_path:
        raise ValueError("Dataset path is required for training")
    
    print(f"Training new model with data from {dataset_path}...")
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    
    # Train the model from scratch
    model = train_model(dataset_path, labels)
    return model, labels

# Function to train the model (called by get_model when needed)
def train_model(dataset_path, labels):
    """Train a new hand sign recognition model"""
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    
    # Data loading
    print("Loading training data...")
    data = []
    target = []
    
    # Track progress
    total_classes = len(labels)
    
    for label_idx, label in enumerate(labels):
        label_path = os.path.join(dataset_path, label)
        print(f"Processing class {label} ({label_idx+1}/{total_classes})")
        
        if not os.path.exists(label_path):
            print(f"Warning: Directory not found for class {label}. Skipping.")
            continue
            
        for img_name in os.listdir(label_path):
            try:
                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                image = cv2.resize(image, IMAGE_SIZE)
                image = image / 255.0  # Normalize
                data.append(image)
                target.append(label_idx)
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    print(f"Loaded {len(data)} images across {total_classes} classes")
    
    # Convert to numpy arrays
    data = np.array(data)
    target_array = np.array(target)
    target = to_categorical(target_array, num_classes=len(labels))
    
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target_array
    )
    
    print(f"Training with {len(x_train)} images, validating with {len(x_val)} images")
    
    # Create model
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(labels), activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=0.0001
    )
    
    # Train model
    print(f"Starting training for up to 15 epochs...")
    model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model


# In[6]:


# Prediction function
def predict_hand_sign(model, image, labels):
    """
    Predict hand sign from image
    
    Args:
        model: Trained TensorFlow model
        image: OpenCV image (BGR format)
        labels: List of class labels
        
    Returns:
        prediction: Predicted class name
        confidence: Confidence score (0-100%)
    """
    # Preprocess the image
    resized_img = cv2.resize(image, IMAGE_SIZE)
    normalized_img = resized_img / 255.0
    
    # Make prediction
    pred = model.predict(np.expand_dims(normalized_img, axis=0), verbose=0)[0]
    
    # Get the predicted class
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class] * 100
    
    return labels[predicted_class], confidence


# In[10]:


def run_webcam(model, labels):
    """Run hand sign recognition using OpenCV webcam interface"""
    cap = cv2.VideoCapture(0)
    
    print("Starting webcam. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Draw rectangle for hand placement
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        rect_size = min(w, h) // 3
        
        cv2.rectangle(
            frame, 
            (center_x - rect_size // 2, center_y - rect_size // 2),
            (center_x + rect_size // 2, center_y + rect_size // 2),
            (0, 255, 0), 2
        )
        
        # Extract hand region for prediction
        hand_region = frame[
            center_y - rect_size // 2:center_y + rect_size // 2,
            center_x - rect_size // 2:center_x + rect_size // 2
        ]
        
        if hand_region.size > 0:
            # Add slight delay to avoid constant predictions which can slow down the UI
            sign, confidence = predict_hand_sign(model, hand_region, labels)
            
            # Display prediction on frame
            cv2.putText(
                frame, 
                f"{sign}: {confidence:.1f}%", 
                (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2
            )
        
        # Display the frame
        cv2.imshow('Hand Sign Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




