# 🔥 Fire Detection Web App

A real-time fire detection web application built using YOLOv8 and Streamlit. This app can identify fire in images or video streams and display the results interactively in a browser.

---

## 🚀 Features

- Detects fire in uploaded images or video streams
- Powered by YOLOv8 (Ultralytics)
- Pre-trained model (`best.pt`) for high accuracy
- User-friendly interface with Streamlit
- Visual feedback with bounding boxes around detected fire

---

## 🛠️ Tech Stack

- **Python** – Programming Language
- **Streamlit** – Web Interface
- **Ultralytics** – YOLOv8 Model Framework
- **OpenCV** – Image/Video Processing
- **PyTorch** – Deep Learning Backend

---

## 📁 Project Structure

fire-detection/
│
├── app.py               → Streamlit Web App  
├── best.pt              → Trained YOLOv8 model file  
├── main.py              → Optional CLI/testing script  
├── fire-detection.ipynb → Jupyter Notebook for training/testing  
├── fire.jpeg            → Sample input image  
├── requirements.txt     → Python dependencies  
└── venv/                → Virtual environment folder

---