# YOLO-Based Detection, Localization, and Classification of Kidney Stones from CT Images

## 📌 Overview

This project uses the YOLO object detection algorithm to automatically detect kidney stones in CT scan images. The model identifies the presence of stones, localizes them using bounding boxes, determines their position (left or right kidney), estimates their size, and classifies them into different categories based on image characteristics.

This project was developed as my Final Year Project for the Bachelor of Engineering in Computer Science.

---

## ✨ Features

- Detects kidney stones from CT scan images
- Localizes stones using bounding boxes
- Identifies left or right kidney location
- Calculates stone size
- Classifies stones into different categories
- Displays detection confidence
- Saves prediction results

---

## 🛠️ Tech Stack

- Python
- YOLO (Ultralytics)
- OpenCV
- PyTorch
- NumPy
- Pillow

---

## 📂 Project Structure

```
project/
│
├── dataset/
├── models/
├── runs/
├── images/
├── predict.py
├── train.py
├── requirements.txt
├── README.md
└── best.pt
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/kidney-stone-detection.git
```

Move into the project directory

```bash
cd kidney-stone-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python predict.py
```

---

## 📊 Model

- Model: YOLO
- Framework: Ultralytics
- Language: Python

---

## 📈 Output

The model provides:

- Kidney stone detection
- Bounding box localization
- Confidence score
- Stone location
- Stone size estimation

---

## 📷 Sample Output

Add screenshots of your prediction results here.

Example:

![Prediction](images/output.png)

---

## 📚 Future Improvements

- Support DICOM images
- Real-time clinical deployment
- Improve detection accuracy with larger datasets
- Develop a web application for prediction

---

## 👩‍💻 Author

**Aarthi A**

LinkedIn: https://www.linkedin.com/in/yourprofile

GitHub: https://github.com/yourusername
