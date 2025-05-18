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


# In[2]:


dataset_path = "C:/Users/Tisha Verma/Desktop/UML Project - Hand Signs/Data"  # Change this to your dataset path
labels = sorted(os.listdir(dataset_path))


# In[3]:


data, target = [], []
image_size = (64, 64)  # Resize images for uniformity


# In[4]:


# Load and preprocess data
for label in labels:
    label_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(label_path):
        try:
            img_path = os.path.join(label_path, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize
            data.append(image)
            target.append(labels.index(label))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")


# In[5]:


# Convert data to numpy arrays
data = np.array(data)
target = to_categorical(target, num_classes=len(labels))


# In[6]:


# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=42)


# In[7]:


print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")


# In[8]:


# CNN Model Creation
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])


# In[9]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[10]:


# Model Training
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_val, y_val))


# In[11]:


# Webcam Detection Function
def predict_image(image):
    resized_img = cv2.resize(image, image_size) / 255.0
    pred = model.predict(np.expand_dims(resized_img, axis=0))
    return labels[np.argmax(pred)]


# In[15]:


# Streamlit Integration
import streamlit as st

st.title("Hand Sign Detection System")
run = st.button("Start Webcam")
if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not accessible!")
            break

        prediction = predict_image(frame)
        cv2.putText(frame, prediction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# In[ ]:




