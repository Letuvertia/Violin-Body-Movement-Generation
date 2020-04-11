# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 00:50:23 2019

@author: user
"""

import numpy as np
import os
import csv
import tensorflow as tf
import matplotlib.pyplot as plt

def getSongID(elem):
    return elem[0]

def getMusicAddressAndSorting(music_location):
    music_address = []
    for dirPath, dirNames, fileNames in os.walk(music_location):
        for f in fileNames:
            music_address.append( [ int(f.split('.')[0]), dirPath+str(f) ] )
    music_address.sort(key=getSongID)
    
    return music_address

anno_len = []
def readData(music_location, annotation_location, import_song, seqLength, fps):
    
    ## reading arousal annotations
    print('Importing Annotations')
    Y = []
    file = open(annotation_location, 'r')
    annotation_data = file.readlines()[1:]
    for count_song, song_anno in enumerate(annotation_data):
        if count_song >= import_song:
            break
        oneSongAnnoList = []
        for anno in song_anno.split(',')[1:]:
            oneSongAnnoList.append([float(anno)])
        songAnnoLength = len(song_anno.split(',')[1:])
        
        # turn into time sequence (non-overlapping)
        # (Annotations, 1) => (Annotations/seqLength, seqLength, 1)
        for i in range( int(songAnnoLength/seqLength) ):
            Y.append( oneSongAnnoList[ i*seqLength : i*seqLength+seqLength ] )
        
        anno_len.append(songAnnoLength)
    
    
    ## reading audio spectrogram
    print('Importing MelSpectrogram')
    X=[]
    for count_song, song_ID_address in enumerate(getMusicAddressAndSorting(music_location)):
        if count_song >= import_song:
            break
        
        
        print("importing " + str(song_ID_address[0]) + '.csv ' + str(int(count_song/len(anno_len)*100)) + '%')
        oneVidAudio = []
        oneSongMeanedMel = []
        
        with open( song_ID_address[1], newline='') as csvfile:
            rows = csv.reader(csvfile)
            
            # cut rows of all zero in the beginning
            cut_zero = 0
            for row in rows:
                if float(row[0]) != 0.:
                    break
                cut_zero += 1
            
            for rowC, row in enumerate(rows):
                if rowC < cut_zero:
                    continue
                tmp=[]
                for index in row:
                    tmp.append([float(index)])
                oneVidAudio.append(tmp)
            
            
            ## mean and normalize a 500ms segment
            # annotation start at 15 secs = 900 th frame
            # a 500ms segment = time t+-250ms  e.g. [14.75~15.25, 15.25~15.75, ..........]
            
            start_frame = 15*60 
            for annoC in range(anno_len[count_song]):
                frameNow = start_frame + annoC*30 # shift 0.5sec = 30 frame
                melspec = oneVidAudio[ int(frameNow-fps/4) : int(frameNow+fps/4) ]
                while len(melspec) != fps/2:
                    melspec.append(melspec[-1])
                
                # mean
                melspec = np.asarray(melspec)
                mean_spectrum = np.sum(melspec, axis=0)/melspec.shape[0]
                
                # standardize
                mean = (np.sum(mean_spectrum, axis=0)/mean_spectrum.shape[0]).tolist()[0]
                standardV = np.std(mean_spectrum, axis=0).tolist()[0] + 0.000000001
                mean_spectrum = mean_spectrum.tolist()
                for i in range(len(mean_spectrum)):
                    mean_spectrum[i][0] = (mean_spectrum[i][0]-mean)/standardV
                
                oneSongMeanedMel.append(mean_spectrum)
        
        # turn into time sequence (non-overlapping)
        for i in range( int(anno_len[count_song]/seqLength) ):
            X.append( oneSongMeanedMel[ i*seqLength : i*seqLength+seqLength ] )
    
    return X, Y
    

def splitData(X, Y, song, seqLength):
    vidForVal_FrameCount = len(X)
    for i in range(song):
        vidForVal_FrameCount -= int(anno_len[-(i+1)]/seqLength)
    X_train = X[:vidForVal_FrameCount]
    Y_train = Y[:vidForVal_FrameCount]
    X_val = X[vidForVal_FrameCount:]
    Y_val = Y[vidForVal_FrameCount:]
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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Conv2D, Reshape, BatchNormalization
from keras.layers import TimeDistributed, Bidirectional, GRU, MaxoutDense
def buildModel():
    model = Sequential()
    
    model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', input_shape=(seqLength,64,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))
    
    model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))
    
    model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))
    
    model.add(Flatten())
    model.add(Reshape( (30,64*8) ))
    
    model.add(TimeDistributed(Dense(8, activation=None)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(GRU(8, return_sequences=True)))
    model.add(Dropout(0.5))
    
    model.add(TimeDistributed(MaxoutDense(1, nb_feature=8)))
    model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])

    model.summary()
    return model

if __name__ == '__main__':
    
    ## parameter
    model_name = 'arousalNetwork_model' 
    music_location = "<insert data path>/Emotion_Audio_Csv/"
    annotation_location = "<insert data path>/Emotion_Arousal_Csv/arousal.csv"
    
    import_song = 1802
    seqLength = 30
    fps = 60
    batch_size = 128
    epochs = 500
    vidForVal = 100
    
    ## reading data
    X, Y = readData(music_location, annotation_location, import_song, seqLength, fps)
    X_train, Y_train, X_val, Y_val = splitData(X, Y, vidForVal, seqLength)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train = np.asarray(X_train), np.asarray(Y_train)
    X_val, Y_val = np.asarray(X_val), np.asarray(Y_val)
    
    ## building model and training
    model = buildModel()
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), callbacks=[callback])
    model.save(model_name + '.h5')
    
    ## draw loss
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    acc = history.history.get('acc')
    val_acc = history.history.get('val_acc')
    
    plt.figure(0,figsize=(8,6))
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
    plt.savefig(model_name + '_Loss.png', dpi=300, format='png')

    