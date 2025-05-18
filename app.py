#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import cv2
import numpy as np
import time
import os


# In[2]:


# Import functions from the main module
# Make sure hand_sign_recognition.py is in the same directory
from hand_sign_recognition import get_model, predict_hand_sign, IMAGE_SIZE

st.set_page_config(
    page_title="Hand Sign Recognition",
    page_icon="✋",
    layout="wide"
)


# In[3]:


# App header
st.title("Hand Sign Recognition System")
st.markdown("---")


# In[4]:


# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info(
        "This application uses a deep learning model to recognize hand signs "
        "for letters A-Z in American Sign Language.\n\n"
        "Position your hand in the green box and the app will predict the corresponding letter."
    )
    
    st.header("Instructions")
    st.markdown(
        """
        1. Click the **Start Camera** button to begin
        2. Position your hand in the green rectangle
        3. Hold your hand still for best results
        4. Click **Stop Camera** when finished
        """
    )
    
    st.markdown("---")
    st.caption("© 2025 Hand Sign Recognition Project")


# In[5]:


# Load model (this happens only once when the app starts)
@st.cache_resource
def load_cached_model():
    try:
        model, labels = get_model()
        return model, labels, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False


# In[6]:


# Load model with a progress indicator
with st.spinner("Loading model..."):
    model, labels, model_loaded = load_cached_model()

if not model_loaded:
    st.error("Failed to load the model. Please make sure the model file exists.")
    st.info("If you need to train the model first, run the training script separately.")
    st.stop()


# In[7]:


# UI layout
col1, col2 = st.columns([3, 1])

with col1:
    # Camera feed
    frame_placeholder = st.empty()

with col2:
    # Prediction display
    st.subheader("Prediction")
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    
    # History of last 5 predictions
    st.markdown("### Recent Predictions")
    history_placeholder = st.empty()
    
    # Buttons
    start_button = st.button("Start Camera", use_container_width=True)
    stop_button = st.button("Stop Camera", use_container_width=True)


# In[8]:


# Main app logic
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
    
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if start_button:
    st.session_state.camera_running = True
    
if stop_button:
    st.session_state.camera_running = False


# In[9]:


def update_history_display():
    history_html = ""
    for p, c, t in st.session_state.prediction_history[-5:]:
        # Ensure timestamp is a datetime object
        if isinstance(t, float):
            from datetime import datetime
            t = datetime.fromtimestamp(t)
            
        # Format time as HH:MM:SS
        time_str = t.strftime("%H:%M:%S")
        
        # Use color coding based on confidence
        if c >= 80:
            color = "green"
        elif c >= 50:
            color = "orange"
        else:
            color = "red"
        history_html += f"<div style='margin-bottom:5px;'><span style='color:{color};font-weight:bold;'>{p}</span> ({c:.1f}%) <small>{time_str}</small></div>"
    
    history_placeholder.markdown(history_html, unsafe_allow_html=True)


# In[10]:


# Run the webcam
if st.session_state.camera_running:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera connection.")
            st.session_state.camera_running = False
        else:
            # Reset prediction highlight timer
            last_prediction_time = 0
            last_prediction = ""
            last_confidence = 0
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to get frame from webcam")
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
                
                # Make prediction every few frames
                current_time = time.time()
                if hand_region.size > 0 and current_time - last_prediction_time > 0.5:  # Predict every 0.5 seconds
                    sign, confidence = predict_hand_sign(model, hand_region, labels)
                    last_prediction = sign
                    last_confidence = confidence
                    last_prediction_time = current_time
                    
                    # Add to history with timestamp
                    st.session_state.prediction_history.append((sign, confidence, time.time()))
                    update_history_display()
                
                # Always display the last prediction
                if last_prediction:
                    # Display prediction on frame
                    cv2.putText(
                        frame, 
                        f"{last_prediction}: {last_confidence:.1f}%", 
                        (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2
                    )
                    
                    # Update UI elements
                    prediction_placeholder.markdown(f"<h1 style='font-size:72px;text-align:center;'>{last_prediction}</h1>", unsafe_allow_html=True)
                    
                    # Color code the confidence bar
                    if last_confidence >= 80:
                        bar_color = "green"
                    elif last_confidence >= 50:
                        bar_color = "orange"
                    else:
                        bar_color = "red"
                        
                    confidence_placeholder.progress(int(last_confidence))
                    confidence_placeholder.markdown(f"<p style='text-align:center;color:{bar_color};'>Confidence: {last_confidence:.1f}%</p>", unsafe_allow_html=True)
                
                # Convert color from BGR to RGB for Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                
                # Add slight delay to reduce CPU usage
                time.sleep(0.03)
                
                # Convert history timestamps to datetime objects for display
                for i in range(len(st.session_state.prediction_history)):
                    p, c, t = st.session_state.prediction_history[i]
                    if isinstance(t, float):
                        from datetime import datetime
                        st.session_state.prediction_history[i] = (p, c, datetime.fromtimestamp(t))
            
            cap.release()
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.camera_running = False
else:
    # Display a placeholder image when camera is not running
    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = placeholder_img.shape[:2]
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Click 'Start Camera' to begin"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    
    cv2.putText(placeholder_img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
    
    # Convert to RGB for Streamlit
    placeholder_img = cv2.cvtColor(placeholder_img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(placeholder_img, channels="RGB", use_column_width=True)
    
    # Clear prediction display
    prediction_placeholder.empty()
    confidence_placeholder.empty()


# In[ ]:




