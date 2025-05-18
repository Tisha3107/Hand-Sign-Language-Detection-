#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


# Path to your dataset
dataset_path = "C:/Users/Tisha Verma/Desktop/UML Project - Hand Signs/Data"
labels = sorted(os.listdir(dataset_path))
print(f"Found {len(labels)} classes: {labels}")


# In[3]:


# Data loading and preprocessing with improved error handling
data = []
target = []
image_size = (128, 128)  # Increased image size for better feature extraction
failed_images = 0


# In[4]:


# Load and preprocess data with progress tracking
total_images = sum([len(os.listdir(os.path.join(dataset_path, label))) for label in labels])
processed_images = 0


# In[5]:


print("Loading and preprocessing images...")
for label_idx, label in enumerate(labels):
    label_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(label_path):
        try:
            img_path = os.path.join(label_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                failed_images += 1
                continue
                
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize
            data.append(image)
            target.append(label_idx)
            
            processed_images += 1
            if processed_images % 100 == 0 or processed_images == total_images:
                print(f"Processed {processed_images}/{total_images} images")
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            failed_images += 1

print(f"Completed processing. Failed to load {failed_images} images.")


# In[6]:


# Convert data to numpy arrays
data = np.array(data)
target_array = np.array(target)
target = to_categorical(target_array, num_classes=len(labels))

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=42, stratify=target_array)

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")


# In[7]:


# Check class distribution
train_class_counts = np.sum(y_train, axis=0)
val_class_counts = np.sum(y_val, axis=0)

print("Training class distribution:")
for i, label in enumerate(labels):
    print(f"{label}: {train_class_counts[i]}")
    
print("Validation class distribution:")
for i, label in enumerate(labels):
    print(f"{label}: {val_class_counts[i]}")


# In[8]:


# Data augmentation to improve generalization
data_augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[9]:


# Improved CNN Model with more capacity
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Second convolutional block
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Third convolutional block
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Fully connected layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(labels), activation='softmax')
])


# In[10]:


# Model summary
model.summary()


# In[11]:


# Learning rate scheduler to improve training
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# In[12]:


# Compile model with appropriate parameters
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[16]:


# Train model with data augmentation
print("Training model...")
history = model.fit(
    data_augmentation.flow(x_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=[lr_scheduler, early_stopping]
)


# In[17]:


# Evaluate model
test_loss, test_acc = model.evaluate(x_val, y_val)
print(f"Test accuracy: {test_acc:.4f}")


# In[18]:


# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')


# In[19]:


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()


# In[20]:


# Save model
model.save('hand_sign_model.h5')
print("Model saved as 'hand_sign_model.h5'")


# In[21]:


# Confusion matrix to visualize performance
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# In[22]:


y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)


# In[23]:


plt.figure(figsize=(16, 14))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[24]:


# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels))


# In[25]:


# Improved prediction function
def predict_hand_sign(image):
    # Preprocess the image
    resized_img = cv2.resize(image, image_size)
    normalized_img = resized_img / 255.0
    
    # Make prediction
    pred = model.predict(np.expand_dims(normalized_img, axis=0))[0]
    
    # Get top 3 predictions for confidence display
    top_indices = pred.argsort()[-3:][::-1]
    top_signs = [(labels[i], pred[i] * 100) for i in top_indices]
    
    return top_signs


# In[26]:


# Webcam implementation with Streamlit
def streamlit_app():
    import streamlit as st
    
    st.title("Hand Sign Recognition System")
    st.write("This application recognizes hand signs for letters A-Z.")
    
    run = st.button("Start Webcam")
    stop = st.button("Stop Webcam")
    
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        result_text = st.empty()
        
        while True and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not accessible!")
                break
                
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw rectangle for hand placement
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            rect_size = min(w, h) // 2
            
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
                predictions = predict_hand_sign(hand_region)
                
                # Display top prediction on frame
                top_pred, confidence = predictions[0]
                cv2.putText(
                    frame, 
                    f"{top_pred}: {confidence:.1f}%", 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2
                )
                
                # Display all predictions in Streamlit
                result_text.write(f"### Predictions")
                for sign, conf in predictions:
                    result_text.write(f"{sign}: {conf:.1f}%")
            
            stframe.image(frame, channels="BGR")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()


# In[27]:


# Run Streamlit app when executed directly
if __name__ == "__main__":
    streamlit_app()


# In[ ]:





# In[ ]:




