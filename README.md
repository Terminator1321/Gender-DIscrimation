# Gender-DIscrimation
AI-powered face detection and gender classification model that identifies whether a person is male or female.

=====================================================
  Real-Time Gender Detection using CNN and OpenCV
=====================================================

This project uses a Convolutional Neural Network (CNN) built with Keras/TensorFlow and computer vision techniques from OpenCV to perform real-time gender detection from a webcam feed.

## 📋 Features

-   **Real-time Face Detection**: Uses a Haar Cascade classifier to locate faces in the video stream.
-   **Gender Classification**: Employs a trained Keras model to classify the detected face as 'man' or 'woman'.
-   **Confidence Score**: Displays the prediction's confidence percentage on the screen.
-   **Simple Interface**: Visualizes the output directly on the webcam feed with bounding boxes and labels.

---

## 📂 Project Structure

The repository is organized into several key directories and files:

```

.
├── Data
│   └── faces
│       ├── man/          \# Contains training images of men
│       └── woman/        \# Contains training images of women
│
├── face\_classifier
│   └── haarcascade\_frontalface\_default.xml \# Pre-trained OpenCV model for face detection
│
├── Model
│   └── GenderAI\_train.ipynb \# Jupyter Notebook used for training the model
│
├── Test.py               \# The main Python script to run the real-time detector
└── requirements.txt      \# A list of all necessary Python libraries

````

-   **`Data/`**: This directory contains the image dataset used for training the model. It is structured into sub-folders, one for each class ('man' and 'woman'). A zipped version of this folder should be included in the repository.
-   **`face_classifier/`**: Holds the XML file for the Haar Cascade algorithm, which is a fast and effective object detection method used here to find faces in each frame.
-   **`Model/`**: Contains the saved, trained Keras model (`GD.keras`) and the Jupyter Notebook (`GenderAI_train.ipynb`) that details the model's architecture and training process.
-   **`Test.py`**: This is the core script. It captures video from the webcam, detects faces, preprocesses them, and uses the trained model to predict the gender.
-   **`requirements.txt`**: Lists all the dependencies needed to run the project.

---

## 💿 Dataset

The dataset used for training is located in the `Data/faces` directory, sourced from the **UTKFace dataset**. The images are categorized into `man` and `woman` folders.

You can find the original dataset here: [{https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)](https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset)](https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset)

A pre-processed version of the data is included as a `.zip` file in this repository for convenience.

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   Python 3.7+
-   A webcam connected to your computer.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Terminator1321/Gender-DIscrimation.git
    cd Gender-DIscrimation
    ```

2.  **Unzip the dataset:**
    If you have a `Data.zip` file, unzip it in the root directory.

3.  **Install the required libraries:**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    This will install OpenCV, TensorFlow, NumPy, and other necessary packages.

### How to Run

Execute the `Test.py` script from your terminal:

```bash
python Test.py
````

A window will open showing your webcam feed. When a face is detected, a magenta rectangle will appear around it with the predicted gender and confidence score.

**To stop the program, press the 'q' key.**

-----

## ⚙️ How the Code Works (`Test.py`)

1.  **Load Models**: The script begins by loading the Haar Cascade classifier for face detection and our pre-trained Keras model (`GD.keras`) for gender classification.

2.  **Initialize Webcam**: It starts capturing video from the default webcam (`cv2.VideoCapture(0)`).

3.  **Main Loop**: The script enters a loop to process the video feed frame by frame.

      - **Face Detection**: Each frame is converted to grayscale, and the `detectMultiScale` function is used to find the coordinates `(x, y, w, h)` of all faces.
      - **Preprocessing**: For each face found:
          - A **Region of Interest (ROI)** is extracted from the grayscale frame.
          - The ROI is resized to `128x128` pixels to match the input size of our neural network.
          - Pixel values are normalized to a range of `0-1` by dividing by `255.0`.
          - The image dimensions are expanded to prepare it for the model.
      - **Prediction**: The processed ROI is passed to `model.predict()`. The model outputs a single probability value (due to the final sigmoid activation layer).
      - **Interpret Results**:
          - If the probability is `>= 0.5`, the label is set to 'woman'.
          - If the probability is `< 0.5`, the label is set to 'man'.
          - The confidence score is calculated based on this probability.
      - **Display Output**: A rectangle is drawn around the detected face, and the predicted label with its confidence score is displayed above it using `cv2.putText()`.

4.  **Termination**: The loop breaks and the application closes when the 'q' key is pressed.
