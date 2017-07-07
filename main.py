"""
Keras Cifar-10 Classification
On blog:
1. Introduction to Keras library
2. Describe Cifar-10 with visualization example(matrix 10x5-10)
3. Describe create process of neural network in Keras - simple 2-layer network
4. Short introduction to convolutional nets
5. Describe main results
6. Post on blog
Requirements for the project:
1. kod wczytujący zbiór cifar
2. kod dokonujący wizualizacji zbioru, wyświetlenie kilku losowych obrazków ułożonych w macierz wraz z informacją do jakiej należą kategorii
3. dwie funkcje budujące 2 siecie neuronowe w keras o różnej architekturze (różna ilość warstw, różne funkcje aktywacji, dropout - tak aby pokazać wpływ tych paramterów na ostateczną dokładnośc klasyfikacji)
4. projekt powinien zawierać wykresy pokazujący progres uczenia się sieci na podstawie  training/testing loss and accuracies co N-itearcji
5.wyswietlenie wynków klasyfikacji w postaci confusion matrix
6. plik readme opisujący rozwiązanie min 600 słów w nim dwa
7. projekt powinien być logicznie podzielony na pliki
8. kod powinien być obficie skomentowany
"""

# IMPORT MODULES

import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

# 1. LOADING THE CIFAR-10 DATASET
from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols = train_features.shape
num_test, _, _, _ = train_features.shape
num_classes = len(np.unique(train_labels))

# 2. Here are the classes in the dataset, as well as 10 random images from each

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(train_labels[:]==i)[0]
    features_idx = train_features[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::],(1,2,0))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# Data Pre-processing

train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255
# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

# 3. Define Model

model = Sequential()

# 4. Compile Model

# 5. Fit Model

