## Facial Expression Recognition

### Overview
This project aims to build a real-time facial expression recognition system using Convolutional Neural Networks (CNN) and OpenCV. The model is trained to classify seven different emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Files
cnn.py: Contains the code for defining, compiling, and training the CNN model.
app.py: A script to perform real-time emotion recognition using a webcam.

## Model Architecture
Conv2D Layer with 32 filters and a kernel size of (3, 3)
MaxPooling2D Layer
Conv2D Layer with 64 filters and a kernel size of (3, 3)
MaxPooling2D Layer
Conv2D Layer with 128 filters and a kernel size of (3, 3)
MaxPooling2D Layer
Flatten Layer
Dense Layer with 128 units and ReLU activation
Dropout Layer with a rate of 0.5
Dense Layer with 7 units and Softmax activation

## Expmple
![Uploading Screen Shot 2023-09-07 at 01.00.17.png…]()

<img width="1440" alt="Screen Shot 2023-09-07 at 01 01 59" src="https://github.com/holycabbage/Facial_Expression_Recognition/assets/90731193/95e54c40-9501-493f-a80d-8396ffa9e101">

## References
Emotion-recognition: https://github.com/omar-aymen/Emotion-recognition/tree/master

dataset: FER-2013
link: https://www.kaggle.com/datasets/msambare/fer2013
