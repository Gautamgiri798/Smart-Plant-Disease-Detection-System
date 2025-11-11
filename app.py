import gradio as gr
import cv2
import tensorflow as tf
import numpy as np
from collections import deque

# -----------------------------
# Load Model (Global)
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model("trained_model.keras")
print("Model loaded.")

# -----------------------------
# Class Labels
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
# Global state for stabilization (for streaming only)
# -----------------------------
history = deque(maxlen=5) 

# -----------------------------
# Preprocessing Function (CRITICAL FIX)
# -----------------------------
def preprocess_frame(frame):
    """
    Handles any input frame (RGB, RGBA, Grayscale) from Gradio and
    converts it to the exact (64, 64) BGR format your model was trained on.
    """
    
    # --- 1. Robustly convert to 3-channel RGB ---
    if len(frame.shape) == 2:
        # It's Grayscale (H, W)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 1:
        # It's Grayscale (H, W, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        # It's RGBA (H, W, 4)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    else:
        # It's already 3-channel RGB (H, W, 3)
        frame_rgb = frame

    # --- 2. Convert from RGB to BGR (as model expects) ---
    # Your original script used cv2.VideoCapture, which provides BGR frames.
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # --- 3. Resize to model's input size (64, 64) ---
    # Your original script used (64, 64).
    img_resized = cv2.resize(frame_bgr, (64, 64))

    # --- 4. Normalize and add batch dimension ---
    img_normalized = img_resized / 255.0
    img_normalized = img_normalized.astype(np.float32)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# -----------------------------
# Prediction Function for STREAMING (Webcam)
# -----------------------------
def predict_stream(frame):
    """
    Takes a single RGB frame, predicts, and returns an annotated RGB frame.
    Uses 'history' deque for stabilization.
    """
    if frame is None:
        return None
    
    # 1. Preprocess and predict
    # (preprocess_frame now handles RGB -> BGR conversion)
    preprocessed_img = preprocess_frame(frame)
    prediction = model.predict(preprocessed_img, verbose=0)
    predicted_class = np.argmax(prediction)
    history.append(predicted_class)

    # 2. Stabilize prediction
    if len(history) > 0:
        label_index = max(set(history), key=history.count)
        label = class_names[label_index]
    else:
        label = "Initializing..."

    # 3. Return the original frame and the label text
    # (All cv2.putText and color conversion logic removed)
    return frame, label

# -----------------------------
# Prediction Function for UPLOAD (Single Image)
# -----------------------------
def predict_upload(frame):
    """
    Takes a single RGB frame, predicts, and returns an annotated RGB frame.
    Does NOT use 'history' deque.
    """
    if frame is None:
        return None
    
    # 1. Preprocess and predict
    # (preprocess_frame now handles RGB -> BGR conversion)
    preprocessed_img = preprocess_frame(frame)
    prediction = model.predict(preprocessed_img, verbose=0)
    predicted_class = np.argmax(prediction)
    
    # 2. Get label (no stabilization needed)
    label = class_names[predicted_class]

    # 3. Robustly convert original frame to RGB for display
    # (This logic is still needed so the output image displays correctly)
    if len(frame.shape) == 2:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 1:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    else:
        frame_rgb = frame.copy()

    # 4. Return the RGB frame and the label text
    # (All cv2.putText and color conversion logic removed)
    return frame_rgb, label

# -----------------------------
# Gradio Interface (with Tabs)
# -----------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌱 Real-Time Plant Disease Detection
        This app uses a trained CNN to detect plant diseases.
        Use the tabs below to either start a live webcam feed or upload an image.
        """
    )
    
    with gr.Tabs():
        # --- Tab 1: Live Detection ---
        with gr.TabItem("Live Detection"):
            with gr.Row():
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Webcam Feed"
                )
                webcam_output = gr.Image(label="Prediction")
            
            # --- NEW: Add a Label output for the prediction ---
            webcam_label = gr.Label(label="Result")
            
            webcam_input.stream(
                predict_stream, 
                webcam_input, 
                [webcam_output, webcam_label] # --- UPDATED: Output to both components ---
            )

        # --- Tab 2: Upload Image ---
        with gr.TabItem("Upload Image"):
            with gr.Row():
                upload_input = gr.Image(
                    sources=["upload"],
                    label="Upload a plant image",
                    type="numpy"
                )
                upload_output = gr.Image(label="Prediction")
            
            # --- NEW: Add a Label output for the prediction ---
            upload_label = gr.Label(label="Result")

            upload_input.upload(
                predict_upload, 
                upload_input, 
                [upload_output, upload_label] # --- UPDATED: Output to both components ---
            )
            
    # --- Accordions for extra info ---
    with gr.Accordion("About this App"):
        gr.Markdown("This project uses a TensorFlow/Keras CNN model to classify 38 different plant disease categories in real-time.")
    
    with gr.Accordion("Show all 38 classes"):
        gr.JSON(class_names) 

# -----------------------------
# Launch the App
# -----------------------------
if __name__ == "__main__":
    demo.launch()