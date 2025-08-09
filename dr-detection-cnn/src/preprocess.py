import cv2
import numpy as np

def load_and_preprocess_image(path, size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img