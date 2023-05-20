#!/usr/bin/env python

# coding: utf-8

​

# In[1]:

​

​

# Import the necessary libraries

# finding the lbp images of the resized face

from lbp import lbp_calculated_pixel

import tensorflow as tf

from tensorflow import keras

import cv2

import numpy as np

import matplotlib.pyplot as plt

​

​

def haar(image):

    # Load the pre-trained face detection model

    face_cascade = cv2.CascadeClassifier(r'harrcascade_frontface_default.xml')

​

    # Define a dictionary to map emotion labels to their corresponding index

    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

​

    # Load the image

    img = image

​

    # Convert the image to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

​

    # Detect the faces in the image using the Haar Cascade algorithm

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

​

    # Loop over each detected face

    for i, (x, y, w, h) in enumerate(faces):

        # Crop the image to isolate the face

        face_cropped = gray[y:y+h, x:x+w]

​

        # Resize the face image to match the input shape of the emotion recognition model

        face_resized=cv2.resize(face_cropped,(48,48), interpolation=cv2.INTER_AREA)

        # Get the corresponding emotion label from the dictionary

        emotion_label = emotion_prediction(face_resized)

​

        # Draw a rectangle around the detected face and display the predicted emotion label

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(img, f"Face {i+1}: {emotion_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

​

    # Display the image with the detected faces and their predicted emotions

    cv2.imshow('Emotion Detection', img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return emotion_label

​

​

def emotion_prediction(face_resized):

    # Load the pre-trained emotion detection model

    model = keras.models.load_model('cnnlbpmodel.h5')

​

    # Define the emotion labels

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

​

    # Load the input image

    #img = cv2.imread('face_resized')

    #plt.imshow(img)

    #plt.show()

    '''

    # Preprocess the input image to make it compatible with the model's input format

    img = cv2.resize(img, (48, 48))

    mg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)'''

    #img = np.reshape(face_resized, (1, 48, 48, 1))

    #img = np.reshape(face_resized, (48, 48,1))

​

    #lbp

​

    height, width= face_resized.shape

​

    # for lbp We need to convert RGB image into gray one because gray image has one channel only but already converted .

​

    # Create a numpy array as the same height and width of RGB image

    img_lbp = np.zeros((height, width),np.uint8)

​

    for i in range(0, height):

        for j in range(0, width):

            img_lbp[i, j] = lbp_calculated_pixel(face_resized, i, j)

​

    img_lbp = np.reshape(img_lbp, (1, 48, 48, 1))

​

    # Use the model to generate a prediction for the input image

    predictions = model.predict(img_lbp)

​

    # Postprocess the model's prediction to extract the most likely emotion label

    emotion_label = emotion_labels[np.argmax(predictions)]

​

    # Output the predicted emotion label

    return emotion_label
