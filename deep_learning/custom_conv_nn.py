"""
Deep neural networks for image classification with AHE (Architectural heritage elements) dataset.
The dataset can be dowloaded at
https://www.kaggle.com/datasets/ikobzev/architectural-heritage-elements-image64-dataset

Guidelines:
https://github.com/newbieeashish/Architectural-Heritage-Elements-Prediction/blob/master/Architectural%20heritage%20elements%20Prediction.ipynb
https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras.layers
import keras.losses
import keras.callbacks
import keras.applications
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np

# Configuration
np.set_printoptions(threshold=sys.maxsize, suppress=True)

train_dir = '../assets/hae/train'
test_dir = '../assets/hae/test'

BATCH_SIZE = 32
IMG_SIZE = (128, 128)

# ImageDataGenerator class allows:
# - configure random transformations and normalization operations to be done on the image data during training
# - instantiate generators of augmented image batches (and their labels) via .flow(data, labels)
#   or .flow_from_directory(directory).
# These generators can then be used with the Keras model methods that accept data generators as inputs
# https://www.analyticsvidhya.com/blog/2020/08/image-augmentation-on-the-fly-using-keras-imagedatagenerator/
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical'
)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Adding the 3rd dimensions to Image for tensoring and fitting to the Input layer
IMG_SHAPE = IMG_SIZE + (3,)


def get_custom_model() -> keras.models.Model:
    """
    Builds a custom deep neural network model
    :return: Keras nn model (Functional)
    """

    input = keras.Input(shape=IMG_SHAPE)

    # Conv2D layer creates a convolution kernel that is convolved with the layer input
    # to produce a tensor of outputs.
    # https://keras.io/2.15/api/layers/convolution_layers/convolution2d/
    x = keras.layers.Conv2D(32, 3, activation='relu')(input)

    # Max pooling is a sample-based discretization process. The objective is to down-sample an input
    # representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing
    # for assumptions to be made about features contained in the sub-regions binned.
    # https://www.quora.com/What-is-Max-Pooling-2D
    x = keras.layers.MaxPooling2D(2, padding='same')(x)

    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D(2, padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    output = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(input, output, name='Custom_model')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_set,
        batch_size=64,
        epochs=1,
        validation_data=test_set,
        verbose=True
    )

    return model


def get_pre_trained_model() -> keras.models.Sequential:
    """
    Builds a deep neural network on Keras MobileNetV2 pre-trained model basis
    :return: Keras nn model (Sequential)
    """

    pre_trained_model = keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    pre_trained_model.trainable = False

    model = keras.Sequential([
        pre_trained_model,

        # GlobalAveragePooling2D calculates the average value of each feature map in the input tensor
        # and outputs a tensor that is smaller in size.
        # E.g. the layer calculates the average of each 3x3 feature map, resulting in a 1D tensor with
        # three elements.
        # https://saturncloud.io/blog/understanding-the-difference-between-flatten-and-globalaveragepooling2d-in-keras
        keras.layers.GlobalAveragePooling2D(),

        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ],
        name='Pre_trained_model'
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_set,
        epochs=1,
        validation_data=test_set,
        verbose=True
    )

    return model


def predict(model) -> None:
    """
    Makes predictions with a Keras model
    :param model: Keras model
    """

    pred_classes = np.argmax(model.predict(test_set), axis=1)

    classes_labels = list(test_set.class_indices.items())
    pred_labels = np.array(classes_labels)[pred_classes]

    print(pred_labels[:, 0][:10])


if __name__ == '__main__':
    models = [
        get_pre_trained_model(),
        get_custom_model()
    ]

    for m in models:
        print(m.name)
        predict(m)
        print('==================\n')
