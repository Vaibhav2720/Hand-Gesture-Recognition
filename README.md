Hand Gesture Recognition Using OpenCV and TensorFlow
Table of Contents
Introduction
Features
Requirements
Installation
Model Setup
How to Run the Code
Gesture Classes
How it Works
Contributing
License
Introduction
This project implements real-time hand gesture recognition using a webcam, OpenCV for image processing, and a pre-trained Convolutional Neural Network (CNN) model from TensorFlow/Keras. The model predicts hand gestures from live camera input and displays the detected gesture on the video feed.

Features
Real-time hand gesture recognition using a webcam.
Pre-trained CNN model to classify hand gestures.
Support for multiple gesture classes.
Live video stream with gesture predictions displayed on the frame.
Requirements
Python 3.x
OpenCV
NumPy
TensorFlow / Keras
Installation
Step 1: Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
Step 2: Install Dependencies
Make sure you have Python and the necessary libraries installed:

bash
Copy code
pip install opencv-python-headless numpy tensorflow
Step 3: Set Up the Model
Download or Train Your Model: You need a pre-trained model to recognize hand gestures. If you don’t have one yet, train a CNN model on a hand gesture dataset.

Save Your Model: Save the trained model in the .keras format and update the path to it in the code:

python
Copy code
model = load_model("D:\\HandGestureRecognition\\trained_model\\my_model.keras")
How to Run the Code
Run the Python script:

bash
Copy code
python hand_gesture_recognition.py
The script will automatically open the webcam, and the video stream will display the detected gesture in real time.

Press 'q' to quit the video stream.

Gesture Classes
The model is designed to recognize the following hand gestures:

Palm
I (Index Finger Raised)
Fist
Fist Moved
Thumb
Index
OK
Palm Moved
C
Down
Make sure your trained model is aligned with these gesture classes for proper predictions.

How it Works
Webcam Input: The script captures live video from your webcam.

Frame Preprocessing:

The captured frame is converted to grayscale.
It is resized to 64x64 pixels to match the input size expected by the CNN model.
The image is normalized by dividing the pixel values by 255.
The shape of the frame is adjusted to match the input shape of the model: (1, 64, 64, 1).
Prediction: The pre-trained model predicts the gesture from the processed frame, and the result is displayed on the video feed in real-time.

Contributing
Contributions are welcome! If you want to improve the model or add more gesture classes, feel free to open a pull request.

