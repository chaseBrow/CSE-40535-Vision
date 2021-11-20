import glob
import mahotas as mt
from sklearn.svm import LinearSVC
from sklearn import svm
import cv2
import numpy as np
import os


from sklearn.linear_model import LogisticRegression
#log_model = LogisticRegression(solver='lbfgs', max_iter=1000)


def extract(img):
    textures = mt.features.haralick(img)
    ht_mean  = textures.mean(axis=0)
	return ht_mean

path  = '../input/asl_alphabet_train/asl_alphabet_train'
name = os.listdir(path)


labels   = []
train_features = []


for item in name:
	cur_path = path + "/" + item
	cur_label = item
	i = 1

	for item in glob.glob(cur_path + "/*.jpg"):
		print("Processing img - {} in {}".format(i, cur_label))
        img = cv2.imread(item)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = extract(gray)
        train_features.append(features)
		labels.append(cur_label)
		i = i + 1


clf_svm = LinearSVC(random_state=9) #classifier
# for kernel in ("linear", "poly", "rbf"):
#     clf_svm = svm.SVC(kernel=kernel, gamma=2)
#     clf_svm.fit(train_features, labels)


#clf_svm = svm.SVC(kernel='poly')


clf_svm.fit(train_features, labels) 

test_path = '../input/asl_alphabet_test/asl_alphabet_test'
for file in glob.glob(test_path + "/*.jpg"):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = extract(gray) 
    prediction = clf_svm.predict(features.reshape(1, -1))[0]
	cv2.putText(img, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
	print("Prediction - {}".format(prediction))
	cv2.imshow("Test_img", img)
	cv2.waitKey(0)
