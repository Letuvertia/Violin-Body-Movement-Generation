# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:36:31 2019

@author: user
"""

import skeletonGeneration_Functions as md2
import csv
import json

# =============================================================================
# PARAMETERS
# =============================================================================

'''

1.inputMusic_loc : location of input music, **Please be noted that your input music MUST BE 16 BITS WAV!**
2.songBeat : beats per bar of the input music
3.JSON : output into json file (including all the data)
4.CSV : output into csv file (only including skeleton coordination)
5.ThreeD : decide the output skeleton will be 3D or 2D (using different skeleton templates)
    2D:
        coordinates data with format of coco
        joints include 2 dimension * 10 joints = 20-D skeleton data in one frame
        joints number list : [0,1,2,3,4,5,6,7,8,11]
    3D:
        coordinates data with format of kinect v2 
        joints include 3 dimension * 25 joints = 20-D skeleton data in one frame
        joints number list : [0~24]
'''

inputMusic_loc = "./testingTrackforQuickStart.wav"
songBeat = 3
JSON = False
CSV = True
ThreeD = True


# =============================================================================
# MAIN GENERATION
# =============================================================================

'''
    Data required:
        a. bowing attack labels
        b. finger arrangement
        c. arousal value
        d. downbeat
        
    Skeleton generation procedure steps:
        
        1. Bowing Model for Right Hand
            data required : a. bowing label & b. finger arrangement
        
        2. Fingering Model for Left Hand
            data required : b. finger arrangement
        
        3. Expression Model for Body (TorsoTilt)
            data required : c. arousal value
        
        4. Expression Model for Body (HeadAcc)
            data required : c. arousal value & d. downbeat
'''


# a. bowing attack labels
bowlabel_pred = md2.getBowingPoint_threshold(inputMusic_loc)
bowlabel_pred = md2.fixCNNDelay(bowlabel_pred)

# b. finger arrangement
f0 = md2.getf0_viaYAAPT(inputMusic_loc)
f0_30fps = md2.fpsTransform(f0)
fingerArr = md2.getFingerArrangement_fromf0(f0)
md2.writeNote2MIDIFile(fingerArr, inputMusic_loc)

# c. arousal value
arousal = md2.getArousalAnnotation(inputMusic_loc)

# d. downbeat
downbeat = md2.getDownBeatViaMadmom(inputMusic_loc, songBeat)
downbeat = md2.addArousalMean2Downbeat(arousal, downbeat)



if ThreeD:
    
    ##################
    # 3D version
    ##################
    
    # 1. right hand main skeleton
    skeleton = md2.calculatingRightHandSkeleton_3D(bowlabel_pred, fingerArr)
    skeleton = md2.skeleton2numpyType_3D(skeleton)
    
    # 2. left hand finger movement
    skeleton = md2.addLeftHandPositionMovement_Numpy_3D(skeleton, fingerArr)
    
    # 3. body emotion-induced movement
    skeleton = md2.addBodyMovement_Numpy_3D(skeleton, arousal)
    
    # 4. head emotion-induced movement
    skeleton = md2.addDJShakyHeadMovement_3D(skeleton, arousal, downbeat, songBeat)
    
    # &. smooth the output skeleton
    skeleton = md2.smoothSkeleton_Medfilt(skeleton, window_size=5, dimension=3, joints_num=25)


if not ThreeD:
    
    ##################
    # 2D version
    ##################

    # 1. right hand main skeleton
    skeleton = md2.calculatingRightHandSkeleton(bowlabel_pred, fingerArr)
    skeleton = md2.skeleton2numpyType(skeleton)
    
    # 2. left hand finger movement
    skeleton = md2.addLeftHandPositionMovement_Numpy(skeleton, fingerArr)
    
    # 3. body emotion-induced movement
    skeleton = md2.addBodyMovement_Numpy(skeleton, arousal)
    
    # 4. head emotion-induced movement
    skeleton = md2.addDJShakyHeadMovement(skeleton, arousal, downbeat, songBeat)
    
    # &. smooth the output skeleton
    skeleton = md2.smoothSkeleton_Medfilt(skeleton, window_size=5, dimension=2, joints_num=10)











savefilename = inputMusic_loc.split('/')[-1].split('.')[0] + "_Ske"

# =============================================================================
# OUTPUT INTO JSON (ALL THE DATA)
# =============================================================================
data_total_list = {}
if JSON:
    print('=======OUTPUT INTO JSON FILE=======')
    print('save in name of : ' + savefilename + '.json' )
    
    data_total_list['skeleton'] = md2.skeleton2ListType(skeleton)
    data_total_list['bowingLabel'] = bowlabel_pred
    data_total_list['f0'] = f0_30fps
    data_total_list['fingerArr'] = fingerArr
    data_total_list['arousal'] = arousal
    data_total_list['downbeat'] = downbeat
    
    with open(savefilename + '.json', 'w') as jsonfile:
        json.dump(data_total_list, jsonfile)


# =============================================================================
# OUTPUT INTO CSV (SKELETON)
# =============================================================================
if CSV:
    if not ThreeD:
        print('=======OUTPUT INTO CSV FILE=======')
        print('save in name of : ' + savefilename + '.csv' )
    
        with open(savefilename + ".csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for f in skeleton:
                tmp = []
                for j in f:
                    tmp.append(j[0])
                    tmp.append(j[1])
                writer.writerow(tmp)
    if ThreeD:
        print('=======OUTPUT INTO CSV FILE (3D format)=======')
        print('save in name of : ' + savefilename + '.csv' )
        
        timeMark = md2.getTimeMark()
        
        with open(savefilename + ".csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow( ['SpineBase.x', 'SpineBase.y', 'SpineBase.z', 'state',
                              'SpineMid.x', 'SpineMid.y', 'SpineMid.z', 'state',
                              'SpineShoulder.x', 'SpineShoulder.y', 'SpineShoulder.z', 'state',
                              'Neck.x', 'Neck.y', 'Neck.z', 'state',
                              'Head.x', 'Head.y', 'Head.z', 'state',
                              'ShoulderLeft.x', 'ShoulderLeft.y', 'ShoulderLeft.z', 'state',
                              'ElbowLeft.x', 'ElbowLeft.y', 'ElbowLeft.z', 'state',
                              'WristLeft.x', 'WristLeft.y', 'WristLeft.z', 'state',
                              'HandLeft.x', 'HandLeft.y', 'HandLeft.z', 'state',
                              'HandTipLeft.x', 'HandTipLeft.y', 'HandTipLeft.z', 'state',
                              'ThumbLeft.x', 'ThumbLeft.y', 'ThumbLeft.z', 'state',
                              'ShoulderRight.x', 'ShoulderRight.y', 'ShoulderRight.z', 'state',
                              'ElbowRight.x', 'ElbowRight.y', 'ElbowRight.z', 'state',
                              'WristRight.x', 'WristRight.y', 'WristRight.z', 'state',
                              'HandRight.x', 'HandRight.y', 'HandRight.z', 'state',
                              'HandTipRight.x', 'HandTipRight.y', 'HandTipRight.z', 'state',
                              'ThumbRight.x', 'ThumbRight.y', 'ThumbRight.z', 'state',
                              'HipLeft.x', 'HipLeft.y', 'HipLeft.z', 'state',
                              'KneeLeft.x', 'KneeLeft.y', 'KneeLeft.z', 'state',
                              'AnkleLeft.x', 'AnkleLeft.y', 'AnkleLeft.z', 'state',
                              'FootLeft.x', 'FootLeft.y', 'FootLeft.z', 'state',
                              'HipRight.x', 'HipRight.y', 'HipRight.z', 'state',
                              'KneeRight.x', 'KneeRight.y', 'KneeRight.z', 'state',
                              'AnkleRight.x', 'AnkleRight.y', 'AnkleRight.z', 'state',
                              'FootRight.x', 'FootRight.y', 'FootRight.z', 'state', 'time(min_s_ms)'])
            
            for frameC, f in enumerate(skeleton):
                tmp = []
                for j in f:
                    tmp.append(j[0])
                    tmp.append(j[1])
                    tmp.append(j[2])
                    tmp.append('tracked')
                tmp.append(timeMark[frameC])
                writer.writerow(tmp)
        
            # add stopTimeMark
            stopMark = md2.get_ItIsTimeToStop_Mark_2()
            for i in stopMark:
                writer.writerow(i)
        