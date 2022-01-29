# -*- coding: utf-8 -*-
"""
Reconhecimento de emoções com Deep Learning utilizando TensorFlow, Keras e OpenCV
"""

import cv2
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow

test_dataset = {
  "class_indices":{
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    'Surprise': 6
  }
}

with open('./network_emotions.json', 'r') as json_file:
    json_saved_model = json_file.read()

imagem = cv2.imread('/content/happy7.jpeg')

#cv2_imshow(imagem)

imagem.shape

detector_face = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')

imagem_original = imagem.copy()

deteccoes = detector_face.detectMultiScale(imagem_original)

deteccoes

roi = imagem[75:75 + 178, 128:128 + 178]

#cv2_imshow(roi)

roi.shape

roi = cv2.resize(roi, (48,48))
cv2_imshow(roi)

roi
roi = roi / 255
roi
roi = np.expand_dims(roi, axis=0)
roi.shape

network_loaded = tf.keras.models.model_from_json(json_saved_model)

probabilidades = network_loaded.predict(roi)

previsao = np.argmax(probabilidades)
print('previsao: ',previsao)

test_dataset["class_indices"]