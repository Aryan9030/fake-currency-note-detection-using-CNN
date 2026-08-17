💵 Fake Currency Note Detection Using CNN
📌 Project Overview

Fake Currency Note Detection Using CNN is a Deep Learning-based computer vision project that detects whether a currency note is Genuine or Fake from an input image.

The system uses a Convolutional Neural Network (CNN) to learn visual patterns and features from currency note images. After training, the model can classify a newly provided currency image as Real or Fake.

This project demonstrates the application of Deep Learning, Image Processing, and Computer Vision for automated currency authentication.

🎯 Objectives
Detect fake currency notes using image-based analysis.
Build a CNN model for currency classification.
Automatically extract important visual features from currency images.
Reduce dependency on manual inspection.
Provide a simple and fast currency verification system.
🧠 Technologies Used
Python
TensorFlow
Keras
CNN (Convolutional Neural Network)
OpenCV
NumPy
Matplotlib
Pillow
Streamlit (optional, for web interface)
🔍 How It Works
Currency Note Image
        ↓
Image Preprocessing
        ↓
Resize & Normalize
        ↓
CNN Model
        ↓
Feature Extraction
        ↓
Classification
        ↓
┌───────────────┐
│ Genuine / Fake│
└───────────────┘

The CNN automatically learns visual patterns such as textures, colors, shapes, and other image characteristics from the training dataset.

🚀 Features
📷 Image-Based Detection

Users can upload or provide an image of a currency note for analysis.

🧠 CNN Classification

A CNN model learns the differences between genuine and counterfeit note images.

⚡ Fast Prediction

Once trained, the model can classify new images quickly.

📊 Prediction Result

The system provides a classification such as:

Prediction: Genuine
Confidence: 94.5%

or

Prediction: Fake
Confidence: 91.2%

Confidence scores represent the model's prediction probability and should not be treated as proof that a note is authentic.

📂 Project Structure
Fake-Currency-Detection/
│
├── dataset/
│   ├── genuine/
│   └── fake/
│
├── model/
│   └── currency_cnn.h5
│
├── test_images/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
│
├── app.py
├── requirements.txt
└── README.md
📊 Dataset

The dataset contains images belonging to two classes:

dataset/
│
├── genuine/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── fake/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

The dataset should contain a diverse collection of genuine and counterfeit currency images captured under different lighting conditions, orientations, backgrounds, and image qualities.

The dataset can be divided into:

Training Set
Validation Set
Testing Set
🧪 CNN Architecture

A typical CNN architecture used for this project can contain:

Input Image
     ↓
Convolution Layer
     ↓
ReLU Activation
     ↓
Max Pooling
     ↓
Convolution Layer
     ↓
ReLU Activation
     ↓
Max Pooling
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Dropout
     ↓
Output Layer
     ↓
Genuine / Fake

The model learns increasingly complex image features through multiple convolution and pooling layers.

⚙️ Installation
1. Clone the Repository
git clone https://github.com/yourusername/Fake-Currency-Detection.git
2. Navigate to the Project
cd Fake-Currency-Detection
3. Create Virtual Environment
python -m venv venv

Activate on Windows:

venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
▶️ Training the Model

Run:

python src/train.py

The training process will:

Load currency images.
Resize the images.
Normalize pixel values.
Split the dataset.
Train the CNN.
Validate the model.
Save the trained model.

Example:

Training CNN...
Epoch 1/20
Epoch 2/20
...
Model saved successfully.
🔮 Making Predictions

After training, use:

python src/predict.py

Provide a currency image and the model will return the predicted class.

Example:

Input: currency_note.jpg

Prediction: Genuine
Confidence: 96.34%
🌐 Streamlit Interface

If a Streamlit interface is included:

streamlit run app.py

The application can provide a simple interface where users upload a currency image and receive the model's prediction.

📈 Model Evaluation

The CNN model can be evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

Example:

Accuracy  : 95%
Precision : 94%
Recall    : 96%
F1-Score  : 95%

These values are examples; actual results depend on the dataset and trained model.

💡 Applications

This project can be useful as an educational prototype for:

Banking systems
Retail stores
Cash handling systems
Automated currency screening
Financial security research
Computer vision applications

For real-world financial use, image-only CNN classification should be combined with appropriate physical and security-feature verification rather than used as the sole authentication method.

🔮 Future Enhancements
Support multiple currency denominations.
Detect notes under different lighting conditions.
Add image augmentation.
Use transfer learning with models such as ResNet or EfficientNet.
Build a mobile application.
Add real-time camera detection.
Improve robustness against different camera angles.
Deploy the model as a cloud API.
Add explainable AI to highlight image regions influencing predictions.
⚠️ Limitations
Performance depends heavily on dataset quality.
Poor lighting and blurred images can affect predictions.
A model trained on one currency or denomination may not generalize to others.
Counterfeit notes can vary significantly from those represented in the training data.
CNN predictions should not be considered definitive authentication.
🔐 Responsible Use

This project is intended for education, research, and prototyping. It should not be used as the sole basis for accepting or rejecting currency in real financial transactions without proper validation and regulatory compliance.

👨‍💻 Author

Aryan Nasina

B.Tech — Artificial Intelligence & Data Science

⭐ Project Goal

To develop a Deep Learning-based system that uses Convolutional Neural Networks to analyze currency note images and classify them as genuine or potentially counterfeit.
