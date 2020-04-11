# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:06:10 2019

@author: user
"""
import json
import csv
import os
import numpy as np
import librosa
from scipy.signal import medfilt, butter, filtfilt


# ***************************************************************
# Details see Section 4.1 Data and preprocessing in the paper
# ***************************************************************


# =============================================================================
# FUNCTIONS FOR FIXING SKELETON DATA
# =============================================================================

def fixedSkeletonJson(location, openpose_joint):
    
    skeleton_fixed, skeleton_lengthOfEachVid = [], []
    file_name = []
    
    
    count=0
    for dirPath, dirNames, fileNames in os.walk(location):
        if dirPath == location:
            print("=====Start Fixing Skeleton Data=====")
            continue
        
        file_name.append(dirPath.split("/")[-1])
        
        ## get raw data from json file
        print("importing and processing " + dirPath.split("/")[-1])
        oneVidSke = []
        for f in fileNames:
            file = open(dirPath+"/"+f, "r")
            fileJson = json.loads(file.read())
            arrayJson = fileJson.get('people')[0].get('pose_keypoints_2d')
            arrayJson_cut = []
            for joint in openpose_joint:
                arrayJson_cut.append(arrayJson[joint*3])
                arrayJson_cut.append(arrayJson[joint*3+1])
            oneVidSke.append(arrayJson_cut)
            count += 1
        
        ## standardization origin points in each video
        mp_x, mp_y = 0, 0
        for frame in range(len(oneVidSke)):
            for joint in range(len(openpose_joint)):
                mp_x += oneVidSke[frame][joint*2]
                mp_y += oneVidSke[frame][joint*2+1]
        jointCount = len(oneVidSke)*len(openpose_joint)
        mp_x, mp_y = mp_x/jointCount, mp_y/jointCount
        
        for frame in range(len(oneVidSke)):
            for joint in range(len(openpose_joint)):
                oneVidSke[frame][joint*2] -= mp_x
                oneVidSke[frame][joint*2+1] -= mp_y
        
        ## smoothing on frame level on each joint, using mediam filter
        for joint in range(len(openpose_joint)*2):
            oneJointSke = []
            for frame in range(len(oneVidSke)):
                oneJointSke.append(oneVidSke[frame][joint])
            oneJointSke = medfilt(oneJointSke, 5)
            for frame in range(len(oneVidSke)):
                oneVidSke[frame][joint] = oneJointSke[frame]
                
        skeleton_fixed.append(oneVidSke)
        skeleton_lengthOfEachVid.append(count)
        count=0
        
    return skeleton_fixed, skeleton_lengthOfEachVid, file_name


def write2csvfile(skeleton_fixed, file_name, file_output_loc):
    
    ## write into csv
    for vidCount, name in enumerate(file_name):
        print('writing file '+name+' into csv file')
        with open(file_output_loc + name + '_fixed.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for frame in range(len(skeleton_fixed[vidCount])):
                writer.writerow(skeleton_fixed[vidCount][frame])
                

# =============================================================================
# FUNCTION FOR AUDIO PROCESSING
# =============================================================================
def audio2melSpectrogram2csvfile(music_location, n_mels, hop_length, file_output_loc):
    
    for dirPath, dirNames, fileNames in os.walk(music_location):
        
        for f in fileNames:
            print("Processing: "+f.split('.')[0])
            
            # wav 2 melSpectrogram using librosa library
            data, sr = librosa.core.load(os.path.join(dirPath,f))
            mel = librosa.feature.melspectrogram(data, n_mels=n_mels, hop_length=hop_length)
            
            # melSpectrogram 2 Csv
            print("Turn in to csv: "+f[:-4])
            with open(file_output_loc + str(f[:-4]) + '.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for trackFrameCount in range(mel.shape[1]):
                    trackPerFrame = []
                    for trackFreData in range(mel.shape[0]):
                        trackPerFrame.append(mel[trackFreData][trackFrameCount])
                    writer.writerow(trackPerFrame)
                

# =============================================================================
# FUNCTIONS FOR CALCULATING BOWING ATTACKS
# =============================================================================
                
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

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
            
def calculateBowingAttacks(location, openpose_joint):
    
    bowingAttacks_total = []
    file_name = []
    
    for dirPath, dirNames, fileNames in os.walk(location):
        if dirPath == location:
            print("=====Start Calculating Bowing Attacks=====")
            continue
        
        file_name.append(dirPath.split("/")[-1].split("_")[0])
        
        ## get raw data from json file
        print("processing video no." + dirPath.split("/")[-1].split("_")[0])
        oneVidSke = []
        for f in fileNames:
            file = open(dirPath+"/"+f, "r")
            fileJson = json.loads(file.read())
            arrayJson = fileJson.get('people')[0].get('pose_keypoints_2d')
            arrayJson_cut = []
            for joint in openpose_joint:
                arrayJson_cut.append(arrayJson[joint*3])
                arrayJson_cut.append(arrayJson[joint*3+1])
            oneVidSke.append(arrayJson_cut)
        
        ## standardization origin points in each video
        mp_x, mp_y = 0, 0
        for frame in range(len(oneVidSke)):
            for joint in range(len(openpose_joint)):
                mp_x += oneVidSke[frame][joint*2]
                mp_y += oneVidSke[frame][joint*2+1]
        jointCount = len(oneVidSke)*len(openpose_joint)
        mp_x, mp_y = mp_x/jointCount, mp_y/jointCount
        
        for frame in range(len(oneVidSke)):
            for joint in range(len(openpose_joint)):
                oneVidSke[frame][joint*2] -= mp_x
                oneVidSke[frame][joint*2+1] -= mp_y
        
        ## smooth
        for joint in range(len(openpose_joint)*2):
            oneJointSke = []
            for frame in range(len(oneVidSke)):
                oneJointSke.append(oneVidSke[frame][joint])
            
            # butterworth => turn into velocity => median filter
            oneJointSke_but = lpf(oneJointSke, 5, 30)
            oneJointSke_but_v = turn2Velocity(oneJointSke_but)
            oneJointSke_but_v_med = medfilt(oneJointSke_but_v, 15)
            
            for frame in range(len(oneVidSke)-1):
                oneVidSke[frame][joint] = oneJointSke_but_v_med[frame]
        
        
        
        
        ## calculating bowing attacks,(1 or 0), based on wrist joint velocity of processed skeleton data
        
        
        ## parameters
        # ===============================================================================
        
        # threshold, refer to parameter 'varphi_{ba}' in Section 3.4 in the paper
        gate=1.0
        
        # window range, refer to parameter 'h' in Section 3.4 in the paper 
        forwardWatchSize = 10
        backwardWatchSize = 10
        
        # ===============================================================================
        
        
        
        skeleton_fixed = oneVidSke
        oneJointSke = []
        for frames in skeleton_fixed:
            oneJointSke.append(frames[9])
        
        expandData = []
        for i in range(forwardWatchSize):
            expandData.append(oneJointSke[0])
        for i in oneJointSke:
            expandData.append(i)
        for i in range(backwardWatchSize):
            expandData.append(oneJointSke[-1])
        
        bowingAttacksVid = []
        for i in range(forwardWatchSize, len(expandData)-backwardWatchSize):
            forwardMean = sum(expandData[i-forwardWatchSize:i])/len(expandData[i-forwardWatchSize:i])
            backwardMean = sum(expandData[i:i+backwardWatchSize])/len(expandData[i:i+forwardWatchSize])
            
            if (forwardMean >= gate and backwardMean <= -gate) or (forwardMean <= -gate and backwardMean >= gate):
                bowingAttacksVid.append(1.)
            else:
                bowingAttacksVid.append(0.)
        
        bowingAttacks_total.append(bowingAttacksVid)
    
    
    
    return bowingAttacks_total, file_name


def write2txt(bowingAttacks, file_name, file_output_loc):
    
    ## write into txt
    for vidCount, name in enumerate(file_name):
        print('writing video no.'+name+' into txt file')
        with open(file_output_loc + name + '_label_bowingAttacks.txt', 'w', newline='') as txtfile:
            for i in range(len(bowingAttacks[vidCount])):
                txtfile.write(str(bowingAttacks[vidCount][i])+'\n')


if __name__== "__main__":
    
    '''
    DATA REQUIREMENTS:
        
    *all the preprocessing below are done by dataPreProcessing.py (except the part of 'Video => Photo => Skeleton Coordinations' in 1.2 output)
    
    
    
    1. Bowing attack network : 
        Data source : URMP dataset http://www2.ece.rochester.edu/projects/air/projects/URMP.html
        
        1.1 input: Mel-spectrograms
            Audio("./Violin_Audio_Wav/", directly from URMP) => Mel-spectrograms("./Violin_Audio_Csv/")
        
        1.2 output: Bowing Attack Labels
            Video(directly from URMP) => Photos => Skeleton Coordinations extracted by Openpose("./Violin_Skeleton_Json_unfixed/") => 
            Normalized and Smoothed Skeleton Coordination("./Violin_Skeleton_Csv_fixed/") => Bowing Attack Labels("./Violin_BowingAttacks_Txt/")
    

    2. Arousal Network : data requirements
        Data source : DEAM dataset http://cvml.unige.ch/databases/DEAM/
        
        2.1 input: Mel-spectrograms
            Audio("./Emotion_Audio_Wav/", directly from DEAM) => Mel-spectrograms("./Emotion_Audio_Csv/")
        
        2.2 output: Arousal Attack Labels
            Arousal Attack Labels(directly from DEAM)
               
    '''
    
    openpose_joint=[0,1,2,3,4,5,6,7,8,11] # List of chosen 10 joints of the COCO body model
    skeleton_location = "<insert raw data path>/Violin_Skeleton_Json_unfixed/" # Raw Json files location
    audio_bowing_loc = "<insert raw data path>/Violin_Audio_Wav/" # Raw Audio for Bowing Attacks Network
    audio_arousal_loc = "<insert raw data path>/Emotion_Audio_Wav/" # Raw Audio for Arousal Network


    # fix skeleton data
    skeleton_fixed, skeleton_Length, file_name_ske = fixedSkeletonJson(skeleton_location, openpose_joint)
    write2csvfile(skeleton_fixed, file_name_ske, file_output_loc='./Violin_Skeleton_Csv_fixed/')

    
    # audio processing
    audio2melSpectrogram2csvfile(audio_bowing_loc, n_mels=128, hop_length=735, file_output_loc='./Violin_Audio_Csv/')
    audio2melSpectrogram2csvfile(audio_arousal_loc, n_mels=64, hop_length=367, file_output_loc='./Emotion_Audio_Csv/')
    

    # calculate bowing attacks
    bowingAttacks, file_name_bow = calculateBowingAttacks(skeleton_location, openpose_joint)
    write2txt(bowingAttacks, file_name_bow, file_output_loc='./Violin_BowingAttacks_Txt/')

    
    
    
    
    
            
    
