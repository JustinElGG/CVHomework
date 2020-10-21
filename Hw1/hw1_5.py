import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
label_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model = load_model('cifar10_vgg16.h5', compile=False)

def show_image():
    print(x_train.shape)
    li = [nn for nn in range(x_train.shape[0])]
    randomIndex = random.sample(li, 10)
    print(randomIndex)
    img_list = np.zeros((128, 1, 3), np.uint8)  # (y, x, c)
    for i in range(len(randomIndex)):
        img = x_train[i]
        img = cv2.resize(img, (128, 128))
        textsize = cv2.getTextSize(label_list[int(y_train[i])], cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0]
        textX = int((img.shape[1] - textsize[0]) / 2)
        cv2.putText(img, label_list[int(y_train[i])], (textX, 110), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1, cv2.LINE_AA)
        img_list = np.hstack((img_list, img))

    cv2.imshow('show me what u got', img_list)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parameter():
    print('+ <<< Hyperparameters >>> ', 55*'-')
    print('| Batch size: 32')
    print('| lr: 0.01')
    print('| Opt: SGD(decay=1e-6, momentum=0.9, nesterov=True)')
    print('+', 80*'-')

def structure():
    model.summary()

def log():
    cv2.imshow('this is the looooooooooooooooooooooog', cv2.imread('log.png'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def inference(num: int):
    img = x_test[num]
    img = np.expand_dims(img, 0)
    result = model.predict(img)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(10), result[0], width=0.4, alpha=0.8, color='blue')
    plt.xticks(range(10), label_list)
    plt.ylabel("Confidence")
    plt.xlabel("Category")
    plt.subplot(1, 2, 2)
    plt.imshow(x_test[num])
    plt.show()
