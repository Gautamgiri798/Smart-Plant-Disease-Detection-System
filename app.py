import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av # Important for handling video frames

# -----------------------------
# App Title and Description
# -----------------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱")
st.title("🌱 Real-Time Plant Disease Detection")
st.write("This app uses a trained CNN to detect plant diseases from your webcam feed.")
st.info("Click 'START' to open your webcam. The prediction will appear on the video feed.")

# -----------------------------
# Load Model (Cached)
# -----------------------------
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    print("Loading model...")
    model = tf.keras.models.load_model("trained_model.keras")
    print("Model loaded.")
    return model

model = load_my_model()

# -----------------------------
# Class Labels (from your original script)
# -----------------------------
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -----------------------------
# Preprocessing Function (from your original script)
# -----------------------------
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Video Transformer Class
# -----------------------------
class PlantDiseaseTransformer(VideoTransformerBase):
    def __init__(self):
        # Use a deque to stabilize predictions over the last N frames
        self.history = deque(maxlen=5) # Same as your original script
        print("Transformer initialized.")

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the av.VideoFrame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Replicate your original logic
        img_flipped = cv2.flip(img, 1) # Mirror view

        # Preprocess and predict
        preprocessed_img = preprocess_frame(img_flipped)
        prediction = model.predict(preprocessed_img, verbose=0)
        predicted_class = np.argmax(prediction)
        self.history.append(predicted_class)

        # Stabilize prediction by taking the most common
        if len(self.history) > 0:
            label_index = max(set(self.history), key=self.history.count)
            label = class_names[label_index]
        else:
            label = "Initializing..."

        # Draw label on the *flipped* frame
        cv2.putText(img_flipped, f"Prediction: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert the modified NumPy array back to an av.VideoFrame
        return av.VideoFrame.from_ndarray(img_flipped, format="bgr24")

# -----------------------------
# Streamlit WebRTC Component
# -----------------------------
webrtc_streamer(
    key="plant-disease-detection",
    video_transformer_factory=PlantDiseaseTransformer,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={  # This is needed for deployment on platforms like Streamlit Cloud
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True,
)

st.sidebar.header("About")
st.sidebar.info("This project is based on your 'smart plant disease detection' work. It uses TensorFlow to classify plant diseases in real-time.")
st.sidebar.header("Class Names")
st.sidebar.expander("Show all 38 classes").json(class_names)