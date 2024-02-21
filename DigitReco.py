import pandas as pd
import numpy as np
import cv2
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
import tensorflow as tf
import pickle
 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.imshow(X_train[0], cmap='gray')
plt.show()

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train[:1000, :]
y_train = y_train[:1000]
X_test = X_test[:100, :]
y_test = y_test[:100]

model = svm.SVC()
model.fit(X_train, y_train)

file = open("DigitModel.model","wb")

pickle.dump(model,file)

y_pred = model.predict(X_test)
 
indexToCompare = 0
 
title = 'True: ' + str(y_test[indexToCompare]) + ', Prediction: ' + str(y_pred[indexToCompare])

def acc(cm):
  one = cm.trace()
  two = cm.sum()
  return one/two

conf = metrics.confusion_matrix(y_test,y_pred)
print(conf)
print(acc(conf))

# plt.title(title)
# plt.imshow(X_test[indexToCompare].reshape(28,28), cmap='gray')
# plt.grid(None)
# plt.axis('off')
# plt.show()

img = cv2.imread("Untitled.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

arr = np.array(gray)
array = arr.reshape(1,-1)

pred = model.predict(array)
print("\n","Prediction of Given Image -> ",pred,end="\n\n")