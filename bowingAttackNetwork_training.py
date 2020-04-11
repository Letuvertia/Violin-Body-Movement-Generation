# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:36:13 2019

@author: user
"""

import numpy as np
import os
import csv
import tensorflow as tf
import matplotlib.pyplot as plt


label_len = []
def readData(music_location, label_location):
    
    ## reading bowing attack labels
    Y = []
    for dirPath, dirNames, fileNames in os.walk(label_location):
        count_vid = 0
        for f in fileNames:
            print("importing "+f)
            
            label= []
            file = open(str(label_location)+str(f), 'r')
            label_data = file.readlines()
            for num_la in label_data:
                if float(num_la.strip('\n')) == 1.0:
                    label.append([1.])
                else:
                    label.append([0.])
            
            count_vid += 1
            Y.append(label)
            label_len.append(len(label))

    
    ## reading melspectrogram
    # origin size = (frames, 128)
    # processed size (network input) = (frames, timestep, 128, 1)
    X=[]
    for dirPath, dirNames, fileNames in os.walk(music_location):
        count_csv = 0
        for f in fileNames:
            oneVidAudio = []
            print("importing "+f)
            with open( str(dirPath)+str(f), newline='') as csvfile:
                rows = csv.reader(csvfile)
                count=0
                for row in rows:
                    tmp=[]
                    for index in row:
                        tmp.append([float(index)])
                    oneVidAudio.append(tmp)
                    count = count + 1
                    if count >= label_len[count_csv]:
                        break
            count_csv = count_csv + 1
            
            for i in range(len(oneVidAudio)-timestep):
                X.append(oneVidAudio[i:i+timestep])
    
    Y_reshape = []
    for vidcount in range(len(Y)):
        for framecount in range(int(timestep/2), label_len[vidcount]-int(timestep/2)):
            Y_reshape.append(Y[vidcount][framecount])
    del Y
    
    return X, Y_reshape

def splitData(X, Y, vid):
    vidForVal_FrameCount = 0
    for i in range(vid):
        vidForVal_FrameCount += label_len[-(i+1)] - (timestep - 1)
    X_train = X[:len(X)-vidForVal_FrameCount]
    Y_train = Y[:len(X)-vidForVal_FrameCount]
    X_val = X[len(X)-vidForVal_FrameCount:]
    Y_val = Y[len(X)-vidForVal_FrameCount:]
    return X_train, Y_train, X_val, Y_val

def shuffle(X,Y):
    np.random.seed(16)
    randomList = np.arange(len(X))
    np.random.shuffle(randomList)
    
    X_shuffled = []
    Y_shuffled = []
    for i in randomList:
        X_shuffled.append(X[i])
        Y_shuffled.append(Y[i])
    return X_shuffled, Y_shuffled

from keras import backend as K
def weighted_binary_crossentropy( y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.0-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * adjust_weight + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(logloss, axis=-1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Conv2D, MaxPooling2D
def buildSmallerModel():
    model = Sequential()
    ## CNN part
    model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(timestep,128,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=8, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=weighted_binary_crossentropy, optimizer="adam", metrics=['accuracy'])

    model.summary()
    return model






if __name__== "__main__":
    
    ## dataset location
    music_location = "<insert data path>/Violin_Audio_Csv_chosen(14pieces)/"
    label_location = "<insert data path>/Violin_BowingAttacks_Txt_chosen(14pieces)/" 
    
    ## parameters
    timestep = 64
    batch_size = 128
    epochs = 500
    
    ## reading data
    X, Y = readData(music_location, label_location)
    X_train, Y_train, X_val, Y_val = splitData(X, Y, 4)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train = np.asarray(X_train), np.asarray(Y_train)
    X_val, Y_val = np.asarray(X_val), np.asarray(Y_val)
    
    ## count weight
    yes, no = 0, 0
    for groundtruth in Y_train:
        if groundtruth[0] == 1.0: yes+=1
        if groundtruth[0] == 0.0: no+=1
    adjust_weight = no/yes

    
    model_name = 'bowingAttackNetwork_model'

    ## building model and training
    model = buildSmallerModel()
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), callbacks=[callback])
    model.save(model_name + '.h5')
    
    ## get loss and accurency
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    acc = history.history.get('acc')
    val_acc = history.history.get('val_acc')
    
    ## write the value of loss and accurency in each epoch into csv
    with open(model_name + "loss&acc.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', "loss", "val_loss", "acc", "val_acc"])
        for i in range(len(loss)):
            writer.writerow( [i+1, loss[i], val_loss[i], acc[i], val_acc[i] ] )
    
    ## draw the plot of loss and accurency
    plt.figure(0, figsize=((8,6)))
    plt.subplot(121)
    plt.plot(range(len(loss)), loss, label='Loss')
    plt.plot(range(len(val_loss)), val_loss, label='Val_Loss')
    plt.title('Loss')
    plt.xlabel("epoch")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.subplot(122)
    plt.plot(range(len(acc)), acc, label='Acc')
    plt.plot(range(len(val_acc)), val_acc, label='Val_Acc')
    plt.title('Accurency')
    plt.xlabel("epoch")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(model_name+'_loss.png', dpi=300, format='png')
        

    
    #'''
