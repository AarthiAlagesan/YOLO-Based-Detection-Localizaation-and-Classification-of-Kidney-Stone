# YOLO-Based Detection, Localization, and Classification of Kidney Stones from CT Scan Images

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

project/
│
├── images/
│   ├── input.png
│   ├── output.png
│
├── results/
│   ├── yolo_pr_curve.png
│   ├── yolo_confusion_matrix.png
│   ├── cnn_accuracy.png
│   ├── cnn_loss.png
│
├── predict.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore



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

## 📷 Sample Outputs

### Input CT Image

![Input CT Image](https://github.com/user-attachments/assets/3c3290d8-de4f-4f3a-a497-6cbddbe9a6a8)

### Kidney Stone Detection(Output)

![Kidney Stone Detection](<img width="956" height="1192" alt="KS552" src="https://github.com/user-attachments/assets/f75f1263-6302-4bef-8ba1-65e5814430ac" />
)

## 🪨 Cropped Kidney Stone Images

### Sample 1

![Cropped Kidney Stone](https://github.com/user-attachments/assets/d58abb21-6c68-4db7-a426-bb439b0a90f6)

### YOLO Training Results

![YOLO Training Results](https://github.com/user-attachments/assets/dde4c330-02c6-4342-a2c5-a9db0b93a397)

## 💻 Terminal Output

![Terminal Output](https://github.com/user-attachments/assets/99b13af7-00d9-4c58-8a69-9bc853100851)



## 📚 Future Improvements

- Support DICOM images
- Real-time clinical deployment
- Improve detection accuracy with larger datasets
- Develop a web application for prediction

---

## 👩‍💻 Author

**Aarthi A**

LinkedIn: https://www.linkedin.com/in/aarthi-alagesan2004/

