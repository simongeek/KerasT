# CIFAR-10 IMAGE CLASSIFICATION WITH KERAS CONVOLUTIONAL NEURAL NETWORK TUTORIAL

## What is Keras?

"Keras is an open source neural network library written in Python and capable of running on top of either [TensorFlow](https://www.tensorflow.org/), [CNTK](https://github.com/Microsoft/CNTK) or [Theano](http://deeplearning.net/software/theano/). 

Use Keras if you need a deep learning libraty that:
* Allows for easy and fast prototyping
* Supports both convolutional networks and recurrent networks, as well as combinations of the two
* Runs seamlessly on CPU and GPU

Keras is compatible with Python 2.7-3.5"[1].

Since Semptember 2016, Keras is the second-fastest growing Deep Learning framework after Google's Tensorflow, and the third largest after Tensorflow and Caffe[2].

## What is Deep Learning?

"Deep Learning is the application to learning tasks of artificial neural networks(ANNs) that contain more than one hidden layer. Deep learning is part of [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) methods based on learning data representations.
Learning can be [supervised](https://en.wikipedia.org/wiki/Supervised_learning), parially supervised or [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning)[3]."

## Project desciption

Simple Youtube presentation what type of visualization is generated:

## What will you learn?




You will learn:

* What is Keras library and how to use it
* What is Deep Learning
* How to use ready datasets
* What is Convolutional Neural Networks(CNN)
* How to build step by step Convolutional Neural Networks(CNN)
* What are differences in model results
* What is supervised and unsupervised learning
* Basics of Machine Learning
* Introduction to Artificial Intelligence(AI)

## Project structure

* 4layerCNN.py -
* 6layerCNN.py -
* Plots - directory with plots
* README.md - project description step by step

## Convolutional neural network


### 6-layer neural network

#### Network Architecture

```
OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####      3   32   32
              Conv2D    \|/  -------------------       896     0.0%
                relu   #####     32   32   32
             Dropout    | || -------------------         0     0.0%
                       #####     32   32   32
              Conv2D    \|/  -------------------      9248     0.4%
                relu   #####     32   32   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     32   16   16
              Conv2D    \|/  -------------------     18496     0.8%
                relu   #####     64   16   16
             Dropout    | || -------------------         0     0.0%
                       #####     64   16   16
              Conv2D    \|/  -------------------     36928     1.5%
                relu   #####     64   16   16
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     64    8    8
              Conv2D    \|/  -------------------     73856     3.1%
                relu   #####    128    8    8
             Dropout    | || -------------------         0     0.0%
                       #####    128    8    8
              Conv2D    \|/  -------------------    147584     6.2%
                relu   #####    128    8    8
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####    128    4    4
             Flatten   ||||| -------------------         0     0.0%
                       #####        2048
             Dropout    | || -------------------         0     0.0%
                       #####        2048
               Dense   XXXXX -------------------   2098176    87.6%
                relu   #####        1024
             Dropout    | || -------------------         0     0.0%
                       #####        1024
               Dense   XXXXX -------------------     10250     0.4%
             softmax   #####          10
```

#### Model
```
model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))



    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
```
Train model:
```
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
cnn_n = base_model()
cnn_n.summary()
```
Fit model:
```
cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),shuffle=True)
```
#### Results

All results are for 50k iteration, learning rate=0.01. Neural networks have been trained at **16 cores and 16GB RAM** on [plon.io](https://plon.io/)

* epochs = 10 **accuracy=75.61%**

![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/59651c4b8c5c480001b146f1)
![Keras Training Loss vs Validation Loss](https://plon.io/files/59651c4b8c5c480001b146f3)

**Confusion matrix result:**
```
[[806   9  39  13  28    4   7   9  61  24]
 [ 14 870   4  10   3    4   7   0  28  60]
 [ 69   1 628  64 122   36  44  19  13   4]
 [ 19   5  52 582 109   99  76  29  14  15]
 [ 13   2  44  46 761   27  38  62   6   1]
 [ 15   1  50 189  69  588  31  48   7   2]
 [  8   3  39  53  52   14 814   4  10   3]
 [ 15   3  31  45  63   29   5 795   2  12]
 [ 61  13   8  10  17    1   4   4 875   7]
 [ 23  52  11  10   7    7   5  12  31 842]]
```
Time of learning process: **1h 45min**


* epochs = 20 **accuracy=75.31%**


![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/59654ea78c5c480001b146f9)
![Keras Training Loss vs Validation Loss](https://plon.io/files/59654ea78c5c480001b146fb)

**Confusion matrix result:**
```
[[810   5  30  22  14    2   9  10  60  38]
 [ 13 862   7   8   3    6   4   7  20  70]
 [ 85   2 626  67  84   44  44  27  12   9]
 [ 39   6  47 581  73  137  50  38  17  12]
 [ 22   1  52  87 744   34  22  64   2   2]
 [ 20   3  40 178  44  639  21  48   2   5]
 [ 12   3  42  55  67   16 782  10   7   6]
 [ 15   2  24  38  59   37   3 810   5   7]
 [ 79  14  10  19   6    4   8   5 827  28]
 [ 25  60   8   9   8    5   2  12  21 850]]
```
Time of learning process: **3h 40min**

* epochs = 50 **accuracy=x%**

![Keras Training Accuracy vs Validation Accuracy]()
![Keras Training Loss vs Validation Loss]()

Confusion matrix result:

Time of learning process: **1h 10min**


* epochs = 100 **accuracy=x%**

![Keras Training Accuracy vs Validation Accuracy]()
![Keras Training Loss vs Validation Loss]()

Confusion matrix result:

Time of learning process: **1h 10min**

### 4-Layer neural network


#### Network Architecture
```
OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####      3   32   32
              Conv2D    \|/  -------------------       896     0.1%
                relu   #####     32   32   32
              Conv2D    \|/  -------------------      9248     0.7%
                relu   #####     32   30   30
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     32   15   15
             Dropout    | || -------------------         0     0.0%
                       #####     32   15   15
              Conv2D    \|/  -------------------     18496     1.5%
                relu   #####     64   15   15
              Conv2D    \|/  -------------------     36928     3.0%
                relu   #####     64   13   13
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     64    6    6
             Dropout    | || -------------------         0     0.0%
                       #####     64    6    6
             Flatten   ||||| -------------------         0     0.0%
                       #####        2304
               Dense   XXXXX -------------------   1180160    94.3%
                relu   #####         512
             Dropout    | || -------------------         0     0.0%
                       #####         512
               Dense   XXXXX -------------------      5130     0.4%
             softmax   #####          10
```
#### Model

```
model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr = 0.1, decay=1e-6, nesterov=True)
```
Train model:
```
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
cnn_n = base_model()
cnn_n.summary()
```
Fit model:
```
cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),shuffle=True)
```

#### Results

All results are for 50k iteration, learning rate=0.1. Neural networks have been trained at **16 cores and 16GB RAM** on [plon.io](https://plon.io/)

* epochs = 10  **accuracy=70.45%**



![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/59637666c0265100013c2c7a)
![Keras Training Loss vs Validation Loss](https://plon.io/files/59637667c0265100013c2c7c)

Confusion matrix result:

Time of learning process: **1h 10min**

* epochs = 20 **accuracy=74.57%**



![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/59639691c0265100013c2c80)
![Keras Training Loss vs Validation Loss](https://plon.io/files/59639691c0265100013c2c82)

Confusion matrix result:

Time of learning process: **2h 15min**

* epochs = 50 **accuracy=75.32%**


![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/5963e88fc0265100013c2c8c)
![Keras Training Loss vs Validation Loss](https://plon.io/files/5963e890c0265100013c2c8e)




Confusion matrix result:

Time of learning process: **5h 45min**


* epochs = 100 **accuracy=x%**


## Results




## Resources

1. [Official Keras Documentation](https://keras.io/)
2. [About Keras on Wikipedia](https://en.wikipedia.org/wiki/Keras)
3. [About Deep Learning on Wikipedia](https://en.wikipedia.org/wiki/Deep_learning)
4. [Tutorial by Dr. Jason Brownlee](http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/)
5. [Tutorial by Parneet Kaur](http://parneetk.github.io/blog/cnn-cifar10/)
6. [Tutorial by Giuseppe Bonaccorso](https://www.bonaccorso.eu/2016/08/06/cifar-10-image-classification-with-keras-convnet/)
7. Open Source on GitHub


## Grab the code or run project in online IDE
* You can [download code from GitHub](https://github.com/simongeek/KerasT)
* You can run the project in your browser
