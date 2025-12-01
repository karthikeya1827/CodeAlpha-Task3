# 🔢 Handwritten Digit Classification using CNN (MNIST)

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset. The model achieves **high accuracy (~98.96%)** on the test set.

---

## 📌 Objective
To accurately recognize digits from grayscale images using a deep learning model trained on the MNIST dataset.

---

## 🧠 Approach
- **Preprocessing**:
  - Normalize pixel values to [0, 1]
  - Reshape images to `(28, 28, 1)`
  - One-hot encode labels
- **Model Architecture**:
  - Conv2D (32 filters, 3×3) + MaxPooling
  - Conv2D (64 filters, 3×3) + MaxPooling
  - Flatten → Dense(128, ReLU) → Dense(10, Softmax)
- **Evaluation**:
  - Accuracy and loss on test data
  - Example predictions

---

## 📁 Dataset
- **Source**: `tensorflow.keras.datasets.mnist`
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Image Shape**: 28×28 grayscale

---

## 📊 Results
- **Final Test Accuracy**: **98.96%**
- **Final Test Loss**: **0.0364**
- **Training History**:
  - Epoch 1: Accuracy 94.29%, Val Accuracy 98.34%
  - Epoch 5: Accuracy 99.34%, Val Accuracy 98.79%
  - Epoch 10: Accuracy 99.67%, Val Accuracy 98.96%

- **Example Predictions**:
- True Labels:      [7 2 1 0 4 1 4 9 5 9]
-  Predicted Labels: [7 2 1 0 4 1 4 9 5 9]

  
---

## 🧰 Libraries Used
- `tensorflow.keras`
- `numpy`

---

## 🚀 How to Run
1. Clone the repo
2. Install dependencies:  
 ```bash
 pip install tensorflow numpy

## Run the script
python mnist_cnn.py

