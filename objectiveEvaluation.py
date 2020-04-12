# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 00:25:51 2019

@author: user
"""

import json
import csv
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from sklearn.metrics import f1_score, accuracy_score
from keras.models import load_model
from keras import backend as K

a = './bowingAttackNetwork_model.h5'


def weighted_binary_crossentropy( y_true, y_pred, adjust_weight=3.521048321048321):
    y_true = K.clip(y_true, K.epsilon(), 1.0-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * adjust_weight + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(logloss, axis=-1)

## predict bowing attacks using threshold ( >0.5 == bowing attack )
def getChangingPoint_threshold(music_loc, firstBow='up', threshold=0.5, model_loc = './Bowing Attack Network_params training/Music2BowingPoint_Paper_Parameter.h5'):
    '''
    music sample rate : 30 fps
    output : a list of 'up' or 'down'
    delay  : 32 frames (start at 33th frames)
    '''
    
    print('=======GET CHANGING POINTS(THRESHOLD)=======')
    print('Predicting Potential Changing Points of ' + music_loc.split('/')[-1].split('.')[0] )
    
    model = load_model(model_loc, custom_objects={"weighted_binary_crossentropy":weighted_binary_crossentropy})
    timestep = 64
    
    test_music, sr = librosa.core.load(music_loc)
    test_music_mel = librosa.feature.melspectrogram(test_music, n_mels=128, hop_length=735)
    
    # change audio data format to fit CNN (frames, timestep, 128, 1)
    test_x = []
    for trackFrameCount in range(test_music_mel.shape[1]):
        trackPerFrame = []
        for trackFreData in range(test_music_mel.shape[0]):
            trackPerFrame.append([test_music_mel[trackFreData][trackFrameCount]])
        test_x.append(trackPerFrame)
    
    test_x_graphic = []
    for i in range(len(test_x)-timestep+1):
        test_x_graphic.append(test_x[i:i+timestep])
    test_x_graphic = np.asarray(test_x_graphic)
    
    # predict
    predLabel = model.predict(test_x_graphic).tolist()
        
    # turn into 1 or 0
    predLabel_OneZero = []
    for label in predLabel:
        if label[0] >= threshold:
            predLabel_OneZero.append(1)
        else:
            predLabel_OneZero.append(0)
    
    return predLabel_OneZero

## predict bowing attacks using likelihood
def getChangingPoint_likelihood(music_loc, firstBowing='up', smooth_window_size=5, model_loc = './bowingAttackNetwork_model.h5'):
    '''
    music sample rate : 30 fps
    output : a list of 'up' or 'down'
    delay  : 32 frames (start at 33th frames)
    '''
    
    print('=======GET BOWING ATTACKS(THRESHOLD)=======')
    print('Predicting Potential Bowing Attacks of ' + music_loc.split('/')[-1].split('.')[0] )
    
    model = load_model(model_loc)
    timestep = 64
    
    test_music, sr = librosa.core.load(music_loc)
    test_music_mel = librosa.feature.melspectrogram(test_music, n_mels=128, hop_length=735)
    
    # change music data format to fit CNN (frames, timestep, 128, 1)
    test_x = []
    for trackFrameCount in range(test_music_mel.shape[1]):
        trackPerFrame = []
        for trackFreData in range(test_music_mel.shape[0]):
            trackPerFrame.append([test_music_mel[trackFreData][trackFrameCount]])
        test_x.append(trackPerFrame)
    
    test_x_graphic = []
    for i in range(len(test_x)-timestep):
        test_x_graphic.append(test_x[i:i+timestep])
    test_x_graphic = np.asarray(test_x_graphic)
    
    # predict
    predLabel = model.predict(test_x_graphic).tolist()
    
    # turn into 1 or 0
    predLabel_p = []
    for p in predLabel:
        predLabel_p.append(np.random.binomial(1, p[0], 1)[0]) ## determine 1 or 0 based on possibility
    predLabel_p = medfilt(predLabel_p, smooth_window_size) # smooth
    
    return predLabel_p.tolist()

def fixCNNDelay(array, delay=32):
    '''
    just add delays of array[0]s to the front of the array in order to fix the delay caused by CNN
    '''
    
    for i in range(delay):
        array.insert(0, array[0])
    
    for i in range(delay-1):
        array.append(array[-1])
    
    return array

def countTypeAmount(predLabel_p):
    start_condition, counter = predLabel_p[0], 0
    labelTypeLength = []
    time = []
    for frameC in range(len(predLabel_p)):
        # last one
        if frameC == len(predLabel_p)-1:
            labelTypeLength.append([start_condition, counter+1])
        # change
        if predLabel_p[frameC] != start_condition:
            labelTypeLength.append([start_condition, counter])
            if start_condition == 1:
                time.append( (int(frameC-1)+int(frameC-counter)) /2 /30 )
            start_condition, counter = predLabel_p[frameC], 1
        else:
            counter += 1
    
    return labelTypeLength, time

def f1_score_writtenByMe(tp, fp, fn):
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    
    f1 = 2*precision*recall/(precision+recall)
    
    return f1

def turn2Velocity(skepos):
    skepos = skepos.tolist()
    skepos_v = []
    for i in range(len(skepos)-1):
        skepos_v.append(skepos[i+1]-skepos[i])    
    return np.asarray(skepos_v)

def lpf(x, cutoff, fs, order=4):
    """
    low pass filters signal with Butterworth digital
    filter according to cutoff frequency

    filter uses Gustafsson’s method to make sure
    forward-backward filt == backward-forward filt

    Note that edge effects are expected

    Args:
        x      (array): signal data (numpy array)
        cutoff (float): cutoff frequency (Hz)
        fs       (int): sample rate (Hz)
        order    (int): order of filter (default 5)

    Returns:
        filtered (array): low pass filtered data
    """
    nyquist = fs / 2
    b, a = butter(order, cutoff / nyquist)
    filtered = filtfilt(b, a, x, method='gust')
    return filtered

## calculate the F1 score of bowing attack time
def calculate_F1Score_time(bowlabel_ground, bowlabel_pred, threshold):
    bowlabel_pred, bowlabel_pred_time = countTypeAmount(bowlabel_pred)
    bowlabel_ground, bowlabel_ground_time = countTypeAmount(bowlabel_ground)
    
    bowlabel_ground_time_l = []
    for i in bowlabel_ground_time:
        bowlabel_ground_time_l.append([i, 1])
    
    tp, fp, fn = 0, 0, 0
    for pred in bowlabel_pred_time:
        found = False
        for truth in bowlabel_ground_time_l:
            if abs(pred-truth[0]) <= threshold and truth[1] == 1:
                tp += 1
                truth[1] = 0
                found = True
                break
        
        if not found:
            fp += 1
    
    fn = len([x for x in bowlabel_ground_time_l if x[1]==1 ])
    
    return f1_score_writtenByMe(tp, fp, fn)
    
def read_GroundTruth_BowingAttack(piece):
    inputGroundT_loc = './Violin_BowingAttacks_Txt/' + piece + '_label_bowingAttacks.txt'
    bowlabel_ground = []
    file = open(inputGroundT_loc, 'r')
    label_data = file.readlines()
    for num_la in label_data:
        if float(num_la.strip('\n')) == 1.0:
            bowlabel_ground.append(1)
        else:
            bowlabel_ground.append(0)
            
    return bowlabel_ground

def read_Pred_BowingAttack(piece, isThreshold, random_times):
    
    bowlabel_pred = []
    
    ## threshold
    if isThreshold:
        inputMusic_loc = './Violin_Audio_Wav/' + piece + '.wav'
        bowlabel_pred = getChangingPoint_threshold(inputMusic_loc)
        bowlabel_pred = fixCNNDelay(bowlabel_pred)
    
    ## random
    if not isThreshold:
        for count in range(random_times):
            inputMusic_loc = './Violin_Audio_Wav/' + piece + '.wav'
            bowlabel_pred_each = getChangingPoint_likelihood(inputMusic_loc)
            bowlabel_pred_each = fixCNNDelay(bowlabel_pred_each)
            bowlabel_pred.append(bowlabel_pred_each)
        
    return bowlabel_pred
    
def calculate_OurMethod_F1Score(mode, pieces_ourMethod, pieces_pred, threshold_lst, n=5):
    
    f1Score_OurMethod = []
    
    for threshold in threshold_lst:
    
        print('*******threshold = '+ str(threshold) + '*******')
        f1Score_OurMethod.append([threshold])
       
        if mode == '>0.5':
            bowlabel_ground_allsong = []
            bowlabel_pred_allsong = []
            
            f1Score_frame_tmp = {}
            f1Score_time_tmp = {}
            for i in range(len(pieces_ourMethod)):
                bowlabel_ground = read_GroundTruth_BowingAttack(pieces_ourMethod[i]) # ground truth
                bowlabel_pred = read_Pred_BowingAttack(pieces_pred[i], isThreshold=True, random_times=n) # pred
                print(len(bowlabel_pred))                
                f1Score_frame_tmp[pieces_pred[i]] = accuracy_score( bowlabel_ground, bowlabel_pred[:len(bowlabel_ground)] )
                f1Score_time_tmp[pieces_pred[i]] = calculate_F1Score_time(bowlabel_ground, bowlabel_pred, threshold)
                
                for l in bowlabel_ground:
                    bowlabel_ground_allsong.append(l)
                for l in bowlabel_pred[:len(bowlabel_ground)]:
                    bowlabel_pred_allsong.append(l)
                
                '''
                # write into csv
                with open("bowingAttack_OurMethod" + pieces[i] + '.csv', "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Track: '+pieces[i], 'Ground Truth', 'A2B'])
                    for frameC in range(len(bowlabel_ground)):
                        writer.writerow(['', bowlabel_ground[frameC], bowlabel_pred[frameC]])
                '''
                        
            
            f1Score_frame_tmp['overall'] = accuracy_score( bowlabel_ground_allsong, bowlabel_pred_allsong )
            f1Score_time_tmp['overall'] = calculate_F1Score_time(bowlabel_ground_allsong, bowlabel_pred_allsong, threshold)
            
            f1Score_OurMethod[-1].append( {'bowing direction':f1Score_frame_tmp, 'bowing attack':f1Score_time_tmp} )
        
        
        if mode == 'random':
            f1Score_frame_tmp = {}
            f1Score_time_tmp = {}
            for i in range(len(pieces_ourMethod)):
                bowlabel_ground = read_GroundTruth_BowingAttack(pieces_ourMethod[i]) # ground truth
                bowlabel_pred = read_Pred_BowingAttack(pieces_pred[i], isThreshold=False, random_times=n) # pred
            
                f1Frame_tmp, f1Time_tmp = [], []
                for count in range(n):
                    f1Frame_tmp.append( accuracy_score( bowlabel_ground, bowlabel_pred[:len(bowlabel_ground)] ) )
                    f1Time_tmp.append( calculate_F1Score_time(bowlabel_ground, bowlabel_pred, threshold) )
                
                f1Score_frame_tmp[pieces_pred[i]] = f1Frame_tmp
                f1Score_time_tmp[pieces_pred[i]] = f1Time_tmp
            
            f1Score_frame_tmp['overall'] = [np.mean(f1Frame_tmp), np.std(f1Frame_tmp)]
            f1Score_time_tmp['overall'] = [np.mean(f1Time_tmp), np.std(f1Time_tmp)]
            
            f1Score_OurMethod[-1].append( {'bowing direction':f1Score_frame_tmp, 'bowing attack':f1Score_time_tmp} )

    return f1Score_OurMethod

def skeleton2bowingAttack(fileJson):
    wrist = []
    for t in fileJson:
        wrist.append(t[1][4])
        
    wrist_but = lpf(wrist, 5, 30)
    wrist_but_v = turn2Velocity(wrist_but)
    wrist_but_v_med = medfilt(wrist_but_v, 15)
    
    cp = []
    gate=1.0
    forwardWatchSize = 10
    backwardWatchSize = 10
    
    expandData = []
    for i in range(forwardWatchSize):
        expandData.append(wrist_but_v_med[0])
    for i in wrist_but_v_med:
        expandData.append(i)
    for i in range(backwardWatchSize):
        expandData.append(wrist_but_v_med[-1])
    
    
    for i in range(forwardWatchSize, len(expandData)-backwardWatchSize):
        forwardMean = sum(expandData[i-forwardWatchSize:i])/len(expandData[i-forwardWatchSize:i])
        backwardMean = sum(expandData[i:i+backwardWatchSize])/len(expandData[i:i+forwardWatchSize])
        
        if (forwardMean >= gate and backwardMean <= -gate) or (forwardMean <= -gate and backwardMean >= gate):
            cp.append(1)
        else:
            cp.append(0)
    
    return cp
            
            
            
def calculate_A2B_F1Score(pieces_A2B, threshold_lst, baselineSke_loc):
    
    f1Score_A2B = []
    for threshold in threshold_lst:
        print('*******threshold = '+ str(threshold) + '*******')
        f1Score_A2B.append([threshold])
        
        bowlabel_ground_allsong = []
        bowlabel_pred_allsong = []
        
        f1Score_frame_tmp = {}
        f1Score_time_tmp = {}
        
        for c, i in enumerate(pieces_A2B):
            a2bskeleton_loc = baselineSke_loc + i + '.json'
            file = open(a2bskeleton_loc, "r")
            fileJson = json.load(file)
            
            # ground truth
            groundT_cp = skeleton2bowingAttack(fileJson[0])
            for l in groundT_cp:
                bowlabel_ground_allsong.append(l)
            
            ## pred
            predict_cp = skeleton2bowingAttack(fileJson[1])
            for l in predict_cp:
                bowlabel_pred_allsong.append(l)
            
            '''
            # wirte into csv
            with open("bowingAttack_A2B" + pieces[c] + ".csv", "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Track: '+pieces[c], 'Ground Truth', 'A2B'])
                for frameC in range(len(groundT_cp)):
                    writer.writerow(['', groundT_cp[frameC], predict_cp[frameC]])
            '''
            
            f1Score_frame_tmp[i] = accuracy_score(groundT_cp, predict_cp)
            f1Score_time_tmp[i] = calculate_F1Score_time(groundT_cp, predict_cp, threshold)
        
        f1Score_frame_tmp['overall'] = accuracy_score(bowlabel_ground_allsong, bowlabel_pred_allsong)
        f1Score_time_tmp['overall'] = calculate_F1Score_time(bowlabel_ground_allsong, bowlabel_pred_allsong, threshold)
        
        f1Score_A2B[-1].append( {'bowing direction':f1Score_frame_tmp, 'bowing attack':f1Score_time_tmp} )
        
    return f1Score_A2B
        

if __name__ == '__main__':
    
    '''
    ===========
    PARAMETERS
    ===========
    
    threshold_lst : used for calculating the bowing attack time between the ground truth data and predicted data
                    tolerable time range, set 0.3s in the paper
    
    * calculating Our Method's F1 score
        pieces_ourMethod : name list of bowing attack ground truth.   e.g. 35-2 ==  /35-2_label_bowingAttacks.txt
        pieces_name : audio name list, used for bowing attack prediction
        
    * calculating Baseline(A2B)'s F1 score
        pieces : json file's name list of skeleton coordination output data
        
        
        
    ================
    FUNCTIONS USAGE
    ================
    
    1. calculate_OurMethod_F1Score(mode, pieces, threshold_lst)
        *mode : 
            '>0.5' : if network output(in range[0, 1]) > 0.5,  bowing attack = 1
            'random' : determine 1 or 0 based on possibility, using function from numpy : np.random.binomial(1, p[0], 1)[0])
        *pieces : change both the list 'pieces_ourMethod' and 'pieces_name'
        *threshold_lst : put in tested thresholds, more than one value is avaliable
    
    2. calculate_A2B_F1Score(pieces, threshold_lst)
        *pieces : change the list 'pieces'
        *threshold_lst : put in tested thresholds, more than one value is avaliable
    
    '''
      
    threshold_lst = [0.3]
    pieces_A2B = ['35_2','36_2','39_1','39_2']
    pieces_ourMethod = ['35-2','36-2','39','39-2']
    pieces_pred = ['AuSep_vn_35_Rondeau_2', 'AuSep_vn_36_Rondeau_2', 'AuSep_vn_39_Jerusalem_1', 'AuSep_vn_39_Jerusalem_2']
    
    ## calculate F1 score for OurMethod and Baseline Method
    f1Score_OurMethod = calculate_OurMethod_F1Score('>0.5', pieces_ourMethod, pieces_pred, threshold_lst)
    f1Score_A2B = calculate_A2B_F1Score(pieces_A2B, threshold_lst, './baseline_Results_TestingTracks/')

    
    
    
    
    
    
    

    
    
    
