import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


import os, glob
import cv2

X = []
y = []

#Load images and the labels
for dir_ in os.listdir("./dataset/"):
	#print(dir_)
	#print(os.path.join('/dataset',dir_))
	for file in glob.glob(os.path.join('./dataset',dir_)+"/*.png"):
		img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
		X.append(img.flatten())
		y.append(dir_)

X = np.array(X)
y = np.array(y)


#Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)

print('Before preprocessing')
print(f"Train: X shape={X_train.shape}, y shape={y_train.shape}")
print(f"Test: X shape={X_test.shape}, y shape={y_test.shape}")

#Scaling the input image pixel value, between 0 and 1
#This is done because ml model performs better with these values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# #Converting the interger labels to vectors (lenght 9)
# le = LabelBinarizer()
# y_train = le.fit_transform(y_train)
# y_test = le.transform(y_test)

print('After preprocessing')
print(f"Train: X shape={X_train.shape}, y shape={y_train.shape}")
print(f"Test: X shape={X_test.shape}, y shape={y_test.shape}")


svm = SVC()

svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

predictions = svm.predict(X_test)

print(classification_report(
	y_test,
	predictions,))

#Saving the model
if not os.path.exists('pretrained model'):
	os.makedirs('pretrained model/')

pickle.dump(svm, open('pretrained model/svm_model.pkl', 'wb'))
print("model saved")