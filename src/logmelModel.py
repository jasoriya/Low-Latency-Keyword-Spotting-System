# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:30:06 2019

@author: shreyans
"""
import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from keras.optimizers import Adam, SGD, Adadelta, Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import keras

def getData():
    trainF = np.load('input/mfccTrainFeatures.npy').reshape(-1,32,26,1)
    testF = np.load('input/mfccTestFeatures.npy').reshape(-1,32,26,1)
    trainLabels = pd.read_csv('input/mfcc_labelsTrain.csv')
    testLabels = pd.read_csv('input/mfcc_labelsTest.csv')

    trainL = trainLabels.values[:, 1:]

    c = np.c_[trainF.reshape(len(trainF), -1), trainL.reshape(len(trainL), -1)]
    np.random.shuffle(c)
    trainF = c[:, :trainF.size//len(trainF)].reshape(trainF.shape)
    trainL = c[:, trainF.size//len(trainF):].reshape(trainL.shape)

    train_labels, val_labels, train_features, val_features = train_test_split(trainL, trainF, train_size = 0.8, random_state=42)
    return train_features, val_features, testF, train_labels, val_labels, testLabels.values[:, 1:]

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(32, 26, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(12, activation='softmax'))
    return model


if __name__=="__main__":
    trainF, valF, testF, trainL, valL, testL = getData()
    model = create_model()
    model.summary()
    #opt = Adam(lr=1e-4)
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    history = History()

    callbacks = [history,
             EarlyStopping(monitor='val_loss', patience=20, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='weights/weights_mfcc.best.hdf5',
             monitor='val_loss', verbose=1,
             save_best_only=True, save_weights_only=True, mode='auto')]

    history = model.fit(trainF, trainL, batch_size=100, epochs=200, verbose=1, validation_data=(valF, valL), callbacks=callbacks)

#    model.load_weights('../weights/weights_299_ir.best.hdf5')
    print("The accuracy of cnn is:", model.evaluate(testF, testL))
