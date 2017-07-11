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

* main.py -
* CNN2.py -

## Convolutional neural network


### Single layer neural network

#### Network Architecture

#### Model
```
model = Sequential()
    model.add(Conv2D(32, (3,3),input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten)
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    sgd = SGD(lr = 0.1, momentum=0.9, decay=1e-6, nesterov=False)
```
#### Results

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
