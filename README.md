# Brain-Tumor-Detection-using-CNN


# 🧠 Brain Tumor Detection

## 📌 Overview

Brain Tumor Detection is a Deep Learning project that classifies Brain MRI images into different tumor categories using a Convolutional Neural Network (CNN). Users can upload an MRI scan through a Flask web application and receive the predicted tumor type along with a confidence score.

---

## ✨ Features

- 🧠 Brain MRI Image Classification
- 📷 MRI Image Upload
- 🤖 CNN-Based Prediction
- 📊 Confidence Score
- 🌐 Flask Web Application
- ⚡ Fast & Accurate Detection

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- Flask
- NumPy
- Pillow
- HTML
- CSS

---

## 📂 Dataset

Dataset Name:

Brain MRI Dataset

Classes:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## 📁 Project Structure

```
Brain-Tumor-Detection/

│── app.py
│── train.py
│── predict.py
│── preprocess.py
│── requirements.txt
│── README.md

├── model/
│     brain_tumor_model.keras

├── dataset/
│     glioma/
│     meningioma/
│     no_tumor/
│     pituitary/

├── uploads/

├── templates/
│     index.html

├── static/
│     style.css

└── screenshots/
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Train Model

```bash
python train.py
```

---

## ▶️ Run Project

```bash
python app.py
```

Open Browser

```
http://127.0.0.1:5000
```

---

## 📊 Output

- Upload MRI Image
- Click **Predict**
- AI Detects Brain Tumor Type
- Displays Prediction Result
- Shows Confidence Score

---

## 🎯 Future Improvements

- MRI Image Segmentation
- Grad-CAM Visualization
- Tumor Size Estimation
- Multi-Class Medical Diagnosis
- Mobile Application
- Cloud Deployment

---

## 👨‍💻 Author

**Faizan Khan**

B.Tech Information Technology

AI | Machine Learning | Data Science | Analytics

---

## 📜 License

MIT License
