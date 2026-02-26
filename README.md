# 🌱 Smart Plant Disease Detection System

A **machine learning-powered plant disease detection app** that identifies diseases in plant leaf images using computer vision and deep learning models. This project aims to provide an accessible demo for early warning and diagnosis of plant health issues, helping farmers and growers save crops and improve yield.

Live Demo:  
👉 https://huggingface.co/spaces/Gautamgiri/Smart-Plant-Disease-Detection-System-Using-Smart-Camera

---

## 🧠 Project Overview

Plant disease detection is crucial for improving crop productivity and reducing agricultural losses. This system uses trained deep learning models (CNN-based) to analyze uploaded leaf images and classify them as **healthy** or **diseased**, offering a simple diagnostic interface powered by a web-based application. :contentReference[oaicite:0]{index=0}

The demo demonstrates end-to-end disease classification with visual upload and prediction results.

---

## 🚀 Key Features

- 📷 Upload leaf images for plant disease analysis  
- 🤖 Deep learning model for disease classification  
- 🧪 Demo interface hosted on Hugging Face Spaces  
- 🛠 Implemented using Python and Jupyter notebooks  
- 📊 Easy to extend with more plant classes and model architectures

---

## 🧩 Tech Stack

- **Python** – core development  
- **TensorFlow / Keras** – deep learning  
- **OpenCV / PIL** – image preprocessing  
- **Jupyter Notebooks** – model training & evaluation  
- **Hugging Face Spaces** – live demo deployment

---

## 📦 Repository Contents

| File | Description |
|------|-------------|
| `Train_plant_disease.ipynb` | Model training and evaluation notebook |
| `Test_Plant_Disease.ipynb` | Inference and testing notebook |
| `app.py` | Flask/Streamlit interface for local testing |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Runtime instructions for deployment |

---

## 🚀 How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/Gautamgiri798/Smart-Plant-Disease-Detection-System.git
   cd Smart-Plant-Disease-Detection-System
   ```
   
2. **Create virtual environment**

  ```bash
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate
  ```

3.**Install dependencies**

  ```bash
  pip install -r requirements.txt
  ```

4.**Run the app**

  ```bash
  python app.py
  ```

5.**Open your browser at http://localhost:5000 to try uploading images and view predictions.**

---

## 📌 Notes

- The system currently functions as a demo/proof-of-concept, not a production model.

- Model accuracy depends on the dataset and preprocessing steps.

- You can retrain models with more images for better performance.

---

## 🎯 Future Improvements

- Add more crop types and diseases

- Convert model to TensorFlow Lite for edge deployment

- Integrate hardware (e.g., camera + Raspberry Pi)

- Provide remedy suggestions upon diagnosis

---

## 📫 Feedback & Collaboration

Contributions, suggestions, and discussions are welcome!
Open an issue or pull request to get involved.

---

## 🔗 Live Demo

👉 Try the model online:
https://huggingface.co/spaces/Gautamgiri/Smart-Plant-Disease-Detection-System-Using-Smart-Camera

---
