from time import sleep
import pickle
import cv2
import numpy as np

file = open("DigitModel.model","rb")
model = pickle.load(file)


img = cv2.imread("Untitled.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

arr = np.array(gray)
array = arr.reshape(1,-1)

print("\nPrediction of Given Image -> ",model.predict(array),end="\n\n")
sleep(1)
