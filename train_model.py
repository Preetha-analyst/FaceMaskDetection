import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import os
import numpy as np
from sklearn.utils import shuffle

data = []
labels = []

categories = ["with_mask", "without_mask"]

for i, category in enumerate(categories):
    path = "data/" + category
    if not os.path.exists(path):
        continue
    for img in os.listdir(path):
        try:
            image = cv2.imread(path + "/" + img)
            image = cv2.resize(image, (224,224))
            data.append(image)
            labels.append(i)
        except Exception as e:
            pass

data = np.array(data)/255.0
labels = np.array(labels)

data, labels = shuffle(data, labels, random_state=42)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=15, validation_split=0.2)

model.save("mask_model.h5")
print("Model Saved!")