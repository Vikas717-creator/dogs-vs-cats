# 🐱🐶 Cats vs. Dogs Image Classifier

A simple yet effective **deep learning-based image classifier** that distinguishes between cats and dogs using a **Convolutional Neural Network (CNN)**. Built with **TensorFlow/Keras**, this project demonstrates how neural networks can be applied to binary image classification problems.

---

## 🚀 Features

- 🧠 Trains a custom **CNN model** from scratch  
- 📊 High accuracy on a balanced cats vs. dogs dataset  
- 🖼️ Predicts on new images passed via CLI  
- 📈 Visualizes training and validation metrics  
- 💾 Saves and reuses trained models  
---

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/your-username/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
### 2. Install dependencies
pip install -r requirements.txt
### 3. Prepare the dataset
Organize your image data into the following structure:
dataset/
├── train/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/

## 🔍 How to Predict
To predict a custom image:

python cats_vs_dogs_cnn.py --mode predict --image images/my_dog.jpg
Output:
Image: images/my_dog.jpg  
Prediction: 🐶 Dog

## 📦 Dependencies
The project requires the following Python packages:
tensorflow
numpy
matplotlib
Pillow

## 🧠 Model Architecture
A simple CNN model:

Input: 150x150 RGB image
Conv2D → MaxPooling
Conv2D → MaxPooling
Conv2D → MaxPooling
Flatten → Dense → Output
Final layer uses sigmoid activation for binary classification.

## 📊 Example Output
Epoch 1/10
Train Accuracy: 93.21%
Validation Accuracy: 91.45%
Model saved at model/cat_dog_model.h5

## 🔮 Future Enhancements
🔁 Implement data augmentation
🧠 Use pre-trained models like VGG16 or ResNet
🖼️ Add real-time webcam predictions
🌐 Deploy as a Streamlit or Flask app
📱 Convert model to TensorFlow Lite for mobile apps
📈 Track metrics using TensorBoard

