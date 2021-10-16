import numpy as np
np.random.seed(40535)
import tensorflow
tensorflow.set_random_seed(10)
import matplotlib.pyplot as plt
import os
import cv2
import keras
from sklearn.model_selection import train_test_split

# Pieces of this code were borrowed from multiple different applications of this data
# Help with loading the data: https://www.kaggle.com/gargimaheshwari/asl-recognition-with-deep-learning
# Preprocessing ideas came from the above source in addition to: https://debuggercafe.com/american-sign-language-detection-using-deep-learning/

train = '../asl_alphabet-train/asl-alphabet-train'
evaluation = '../asl-alphabet-test/asl-alphabet-test'
img_size = 64

# directory should be the path to the folder with the images, use the predefined variables 'train' and 'evaluation'
# The size is for resizing the images.  All images are squares.  The reason we decided to implement this size variable is because
# we found multiple different proposed solutions using image sizes ranging from 64 to 580, with such a large range we wanted room to 
# adjust later in the process.
def load_images(directory, size):
    print("loading images")
    
    images = []
    labels = []

    for i, label in enumerate(labels_set):
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            img = cv2.resize(cv2.read(filepath), (size, size))
            images.append(img)
            labels.append(i)
    images = np.array(images)
    labels = np.array(labels)
    return (images, labels)

labels_set = sorted(os.listdir(train))
images, labels = load_images(directory = train, size = img_size)

if labels_set == sorted(os.listdir(evaluation)):
    x_eval, y_eval = load_images(directory = evaluation, size = img_size)

# Here we are going to use scikit to split our training data into two separate pieces, training and testing (note testing is separate from evaluation)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, stratify=labels)

length = len(labels_set)

trainLen = len(x_train)
testLen = len(y_test)

evaluationLen = len(x_eval)


# This printing images code is borrowed from: https://www.kaggle.com/gargimaheshwari/asl-recognition-with-deep-learning
def print_images(image_list):
    n = int(len(image_list) / len(uniq_labels))
    cols = 8
    rows = 4
    fig = plt.figure(figsize = (24, 12))

    for i in range(len(uniq_labels)):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(image_list[int(n*i)])
        plt.title(uniq_labels[i])
        ax.title.set_fontsize(20)
        ax.axis('off')
    plt.show()

y_train_in = y_train.argsort()
y_train = y_train[y_train_in]
x_train = x_train[y_train_in]

print("Training Images: ")
print_images(image_list = x_train)

y_test_in = y_test.argsort()
y_test = y_test[y_test_in]
x_test = x_test[y_test_in]

print("Testing images: ")
print_images(image_list = x_test)

print("Evaluation images: ")
print_images(image_list = x_eval)
    
print("Preprocessing...")

# One hot encoding
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
y_eval = keras.utils.to_categorical(y_eval)

#normalize RGB
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
x_eval = x_eval.astype('float32')/255.0

print('preprocessing complete.')
