from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("mask_model.h5")

img = cv2.imread("test.png")
img = cv2.resize(img, (224,224))

img = np.array(img)/255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print(pred)

print("Prediction Value:", pred[0][0])

if pred[0][0] > 0.5:
    print("NO MASK")
else:
    print("MASK")