# Tyre_Faulty_Detection
"Developed a deep learning-based system to predict faulty tires in manufacturing using image analysis, enabling proactive process optimization and waste reduction."
# Tire Fault Detection System
<img src="https://github.com/rpjinu/Tyre_Faulty_Detection/blob/main/tyre_project_image.png" width="600">


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-brightgreen)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-yellow)

This project aims to develop an **intelligent system** using **deep learning** to predict faulty tires during the manufacturing process. By analyzing tire images and manufacturing parameters, the system identifies defects, enabling manufacturers to optimize processes, reduce waste, and improve production efficiency.

---
## Features

- **Image-Based Fault Detection**: Uses Convolutional Neural Networks (CNNs) to classify tire images as "Good" or "Faulty."
- **Interactive Web App**: Built with **Streamlit** for easy user interaction.
- **Model Training**: Trained on a dataset of tire images with preprocessing, feature extraction, and hyperparameter tuning.
- **Real-Time Predictions**: Upload a tire image and get instant predictions.
- **Visualization**: Displays the uploaded image with the prediction result.

---

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Data Preprocessing**: OpenCV, PIL, NumPy
- **Visualization**: Matplotlib

---

## Dataset

The dataset consists of tire images categorized into two classes:
- **Good Tires**: Images of tires without defects.
- **Faulty Tires**: Images of tires with manufacturing defects.

---

## How It Works

1. **Data Collection**: Gather historical manufacturing data, including images of good and faulty tires.
2. **Data Preprocessing**: Resize, normalize, and augment images for model training.
3. **Model Training**: Train a CNN model using TensorFlow/Keras.
4. **Model Evaluation**: Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
5. **Web App Deployment**: Use Streamlit to create an interactive web app for real-time predictions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rpjinu/tire-fault-detection.git
   cd tire-fault-detection

##Usage:-
Launch the Streamlit app.

Upload a tire image using the file uploader.

The app will display the image and predict whether it is Good or Faulty.

View the confidence score and visual explanation of the prediction.

##Results:-
Achieved 90%+ accuracy on the test dataset.

Successfully deployed the model as a Streamlit web app for real-time predictions.

##Contact:-
For questions or feedback, feel free to reach out:

Name: Ranjan Kumar Pradhan

Email:jinupradhan123@gmail.com

GitHub: rpjinu
