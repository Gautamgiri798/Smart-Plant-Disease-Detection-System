import gradio as gr
import cv2
import tensorflow as tf
import numpy as np
from collections import deque

# -----------------------------
# Load Model (Global)
# -----------------------------
# Model is loaded once when the script starts
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
# Global state for stabilization
# -----------------------------
# Use a deque to stabilize predictions over the last N frames
history = deque(maxlen=5) # Same as your original script

# -----------------------------
# Preprocessing Function (from your original script)
# -----------------------------
def preprocess_frame(frame):
    # Webcam frames are numpy arrays (H, W, C) in BGR format
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Model expects RGB
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

# -----------------------------
# Prediction Function (This replaces the Transformer class)
# -----------------------------
def predict_disease(frame):
    """
    This function takes a single frame from the webcam (as a numpy array)
    and returns a single annotated frame (as a numpy array).
    """
    if frame is None:
        return None

    # Note: Gradio's webcam input is already mirrored, so no cv2.flip() needed.
    
    # 1. Preprocess and predict
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

    # 3. Draw label on the *original* frame
    # (Gradio component will display this frame)
    cv2.putText(frame, f"Prediction: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 4. Return the annotated frame (in BGR for cv2)
    return frame

# -----------------------------
# Gradio Interface
# -----------------------------

# Use gr.Blocks for a custom layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌱 Real-Time Plant Disease Detection
        This app uses a trained CNN to detect plant diseases from your webcam feed.
        Allow webcam access and point a plant leaf at the camera.
        """
    )
    
    with gr.Row():
        # Input: Webcam, streaming
        webcam_input = gr.Image(
            sources=["webcam"],  # <--- THE FIX IS HERE
            streaming=True,
            label="Webcam Feed"
        )
        
        # Output: Annotated image
        annotated_output = gr.Image(
            label="Prediction"
        )
        
    # This "stream" event connects the input to the function to the output
    # It runs the `predict_disease` function on every frame
    webcam_input.stream(predict_disease, webcam_input, annotated_output)
    
    # Recreate the sidebar elements using Accordions
    with gr.Accordion("About this App"):
        gr.Markdown("This project is based on your 'smart plant disease detection' work. It uses TensorFlow to classify plant diseases in real-time.")
    
    with gr.Accordion("Show all 38 classes"):
        gr.JSON(class_names) # Use JSON component to display the list cleanly

# -----------------------------
# Launch the App
# -----------------------------
if __name__ == "__main__":
    demo.launch() # share=True to get a public link