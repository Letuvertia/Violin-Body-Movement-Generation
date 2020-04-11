# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:11:13 2019

@author: user
"""

from keras.models import load_model
import librosa
import numpy as np
import csv
from scipy.signal import medfilt, filtfilt, butter
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
import math

def getBowingPoint_threshold(music_loc, firstBowing='up', threshold=0.5, model_loc = './bowingAttackNetwork_model.h5'):
    '''
    music sample rate : 30 fps
    output : a list of 'up' or 'down'
    delay  : 32 frames (start at 33th frames)
    '''
    
    print('=======GET CHANGING POINTS(THRESHOLD)=======')
    print('Predicting Potential Changing Points of ' + music_loc.split('/')[-1].split('.')[0] )
    
    model = load_model(model_loc)
    timestep = 64
    
    test_music, sr = librosa.core.load(music_loc)
    test_music_mel = librosa.feature.melspectrogram(test_music, n_mels=128, hop_length=735)
    
    # change music data format to fit CNN (frames, timestep, 128, 1)
    test_x = []
    for trackFrameCount in range(test_music_mel.shape[1]):
        trackPerFrame = []
        for trackFreData in range(test_music_mel.shape[0]):
            tmp=[]
            tmp.append(test_music_mel[trackFreData][trackFrameCount])
            trackPerFrame.append(tmp)
        test_x.append(trackPerFrame)
    
    test_x_graphic = []
    for i in range(len(test_x)-timestep):
        test_x_graphic.append(test_x[i:i+timestep])
    test_x_graphic = np.asarray(test_x_graphic)
    
    # predict
    predLabel = model.predict(test_x_graphic).tolist()
    
    # turn into 1 or 0
    #'''
    predLabel_OneZero = []
    for label in predLabel:
        if label[0] >= label[1]:
            predLabel_OneZero.append(1.)
        else:
            predLabel_OneZero.append(0.)
    
    start_condition, counter = predLabel_OneZero[0], 0
    labelTypeLength = []
    for frameC in range(len(predLabel_OneZero)):
        if frameC == len(predLabel_OneZero)-1:
            labelTypeLength.append([start_condition, counter+1])
        if predLabel_OneZero[frameC] != start_condition:
            labelTypeLength.append([start_condition, counter])
            start_condition, counter = predLabel_OneZero[frameC], 1
        else:
            counter += 1
    
    def bowingChange(bowDirection):
        if bowDirection == 'up':
            return 'down'
        else:
            return 'up'
    
    predLabel_UpDown = []
    for labelType in labelTypeLength:
        if labelType[0] == 0.0:    
            for i in range(labelType[1]):
                predLabel_UpDown.append(firstBowing)
        if labelType[0] == 1.0:
            for i in range(int(labelType[1]/2)):
                predLabel_UpDown.append(firstBowing)
            firstBowing = bowingChange(firstBowing)
            for i in range(labelType[1]-int(labelType[1]/2)):
                predLabel_UpDown.append(firstBowing)
                
    print('Output Label Length : ' + str(len(predLabel_UpDown)) + ' frames')
    print('Delay for 32 frames, start at the 33th frame')
    
    return predLabel_UpDown


def getBowingPoint_likelihood(music_loc, firstBowing='up', smooth_window_size=15, model_loc='./bowingAttackNetwork_model.h5'):
    '''
    music sample rate : 30 fps
    output : a list of 'up' or 'down'
    delay  : 32 frames (start at 33th frames)
    '''
    
    print('=======GET CHANGING POINTS(LIKELIHOOD)=======')
    print('Predicting Potential Changing Points of ' + music_loc.split('/')[-1].split('.')[0] )
    
    model = load_model(model_loc)
    timestep = 64
    
    test_music, sr = librosa.core.load(music_loc)
    test_music_mel = librosa.feature.melspectrogram(test_music, n_mels=128, hop_length=735)
    
    # change music data format to fit CNN (frames, timestep, 128, 1)
    test_x = []
    for trackFrameCount in range(test_music_mel.shape[1]):
        trackPerFrame = []
        for trackFreData in range(test_music_mel.shape[0]):
            tmp=[]
            tmp.append(test_music_mel[trackFreData][trackFrameCount])
            trackPerFrame.append(tmp)
        test_x.append(trackPerFrame)
    
    test_x_graphic = []
    for i in range(len(test_x)-timestep):
        test_x_graphic.append(test_x[i:i+timestep])
    test_x_graphic = np.asarray(test_x_graphic)
    
    # predict
    predLabel = model.predict(test_x_graphic).tolist()
    
    predLabel_p = []
    for p in predLabel:
        predLabel_p.append(np.random.binomial(1, p[0], 1)[0])
    predLabel_p = medfilt(predLabel_p, smooth_window_size) # smooth
    
    # count type and number
    start_condition, counter = predLabel_p[0], 0
    labelTypeLength = []
    for frameC in range(len(predLabel_p)):
        if frameC == len(predLabel_p)-1:
            labelTypeLength.append([start_condition, counter+1])
        if predLabel_p[frameC] != start_condition:
            labelTypeLength.append([start_condition, counter])
            start_condition, counter = predLabel_p[frameC], 1
        else:
            counter += 1
    
    # turn into up and down
    def bowingChange(bowDirection):
        if bowDirection == 'up':
            return 'down'
        else:
            return 'up'
    
    predLabel_UpDown = []
    for labelType in labelTypeLength:
        if labelType[0] == 0.0:    
            for i in range(labelType[1]):
                predLabel_UpDown.append(firstBowing)
        if labelType[0] == 1.0:
            for i in range(int(labelType[1]/2)):
                predLabel_UpDown.append(firstBowing)
            firstBowing = bowingChange(firstBowing)
            for i in range(labelType[1]-int(labelType[1]/2)):
                predLabel_UpDown.append(firstBowing)
    
    print('Output Label Length : ' + str(len(predLabel_UpDown)) + ' frames')
    print('Delay for 32 frames, start at the 33th frame')
    
    return predLabel_UpDown


def getArousalAnnotation(music_loc):
    '''
    music sample rate : 60 fps => 2 fps
    output : a list of arousal value
    delay  : no
    '''
    
    print('=======GET AROUSAL ANNOTATIONS=======')
    print('Predicting Arousal Annotations of ' + music_loc.split('/')[-1].split('.')[0] )
    
    seqLength = 20
    output_fps = 30
    fps = 60
    model = load_model('./arousalNetwork_model.h5')

    # Processing data
    # data format : (frames, 64, 1)
    test_music, sr = librosa.core.load(music_loc)
    test_music_mel = librosa.feature.melspectrogram(test_music, n_mels=64, hop_length=367)
    test_x = []
    for trackFrameCount in range(test_music_mel.shape[1]):
        trackPerFrame = []
        for trackFreData in range(test_music_mel.shape[0]):
            trackPerFrame.append([test_music_mel[trackFreData][trackFrameCount]])
        test_x.append(trackPerFrame)
    
    print('Music Length : ' + str(len(test_x)) + ' frames (60 fps)' ) 
    
    # mean and normalize a 500ms segment (30 frames)
    oneSongMeanedMel = []
    start_frame = int(fps/4)
    epsilon = 0.000000001
    annoLen = int(len(test_x)/output_fps)
    for annoC in range(annoLen):
        frameNow = start_frame + annoC*30 # shift 0.5sec = 30 frame
        melspec = test_x[ int(frameNow-fps/4) : int(frameNow+fps/4) ]
        while len(melspec) != fps/2:
            melspec.append(melspec[-1])
        
        # mean
        melspec = np.asarray(melspec)
        mean_spectrum = np.sum(melspec, axis=0)/melspec.shape[0]
        # standarlize
        mean = (np.sum(mean_spectrum, axis=0)/mean_spectrum.shape[0]).tolist()[0]
        standardV = np.std(mean_spectrum, axis=0).tolist()[0] + epsilon
        mean_spectrum = mean_spectrum.tolist()
        for i in range(len(mean_spectrum)):
            mean_spectrum[i][0] = (mean_spectrum[i][0]-mean)/standardV
        
        oneSongMeanedMel.append(mean_spectrum)
    
    # input format : (secs*2, seqLength, 64, 1)
    X = []
    for i in range(annoLen-seqLength+1):
        X.append( oneSongMeanedMel[ i : i+seqLength ] )
    X = np.asarray(X)
    
    predAnno = model.predict(X).tolist()
    predAnno_reshape = []
    for i in range(annoLen):
        predAnno_reshape.append([])
        
    for i, seqLengthPlot in enumerate(predAnno):
        for j, arousal in enumerate(seqLengthPlot):
            predAnno_reshape[i+j].append(arousal[0])
    
    predAnno_final = []
    for i in predAnno_reshape:
        predAnno_final.append( sum(i)/len(i) )
    
    ## music emotion
    arousal = []
    for i in range( int(output_fps/4) ):
        arousal.append(float(predAnno_final[0]))
        
    for acount, a in enumerate(predAnno_final):
        if acount+1 == len(predAnno_final):
            break
    
        twoAdis = float(predAnno_final[acount+1]) - float(predAnno_final[acount])
        for i in range( int(output_fps/2) ):
            value = arousal[-1] + twoAdis / int(output_fps/2)
            arousal.append(value)
    
    for i in range( int(output_fps/2) - int(output_fps/4) ):
        arousal.append(float(predAnno_final[-1]))
        
    print('Output Annotations Length : ' + str(len(arousal)) + ' frames')
    print('P.S.  ' + str(len(arousal)) + ' frames = int( AudioLength(secs)/0.5(secs) ) * 15(frame per 0.5s)')
    
    return arousal


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


def getf0_viaYAAPT(music_loc):
    
    signal = basic.SignalObj(music_loc)
    pitch = pYAAPT.yaapt(signal, **{'f0_max' : 2600.0})
    
    pitch_interp = pitch.samp_interp
    pitch_interp = lpf(pitch_interp, 15, 100)
    pitch_interp = medfilt(pitch_interp, 45)
    
    return pitch_interp.tolist()


def frequency2fingerArrangement(pitchList, fingerArrangementList="./Violin_fingerArrangement_list.csv"):
    '''
    calculate finger arrangement based on frenquency extracted from audio input via YAAPT algorithm
    '''
    
    # get violin finger arrangement total list
    music_transfer = []
    with open(fingerArrangementList, mode='r', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            music_transfer.append(row)
    
    time_pitch=[]
    for timeC, time_frequence in enumerate(pitchList):
        pg = True
        now = []
        now.append(timeC*0.01)
        for loop in range(1, len(music_transfer)):
            if time_frequence >= float(music_transfer[loop-1][5]) and time_frequence < float(music_transfer[loop][5]):
                now.append(loop)
                pg = False
        if(pg):
            now.append('-1')
        time_pitch.append(now)
       
    def LeftHand(a:str):
        rNum = []
        last = 0
        for ch in range(len(a)):
            if(a[ch] == '-'):
                str1 = a[last:ch]
                last = ch + 1
                rNum.append(int(str1))
        return rNum
    
    output = []
    pos_now = 1
    for dt in range(len(time_pitch)):
        if(time_pitch[dt][1] == '-1'):
            output.append('Stop 0-0-0')
            continue
        bDt = True
        for loop in range(1,5):
            rNum = LeftHand(music_transfer[int(time_pitch[dt][1])][loop])
            if(len(rNum) == 0):
                break
            if(rNum[1] == pos_now):
                output.append(music_transfer[int(time_pitch[dt][1])][0] + ' ' + str(rNum[0]) + '-' + str(rNum[1]) + '-' + str(rNum[2]))
                bDt = False
        if(bDt):
            for loop in range(1,5):
                rNum = []
                rNum = LeftHand(music_transfer[int(time_pitch[dt][1])][loop])
                if(len(rNum) == 0):
                    break
                if(rNum[2] == 0):
                    output.append(music_transfer[int(time_pitch[dt][1])][0] + ' ' + str(rNum[0]) + '-' + str(rNum[1]) + '-' + str(rNum[2]))
                    now_pos = rNum[1]
                    bDt = False
                    break
                elif(rNum[2] == 1):
                    output.append(music_transfer[int(time_pitch[dt][1])][0] + ' ' + str(rNum[0]) + '-' + str(rNum[1]) + '-' + str(rNum[2]))
                    now_pos = rNum[1]
                    bDt = False
                    break
    
    return output


def fpsTransform(data, inputFps=100, output_Fps=30):
    '''
    just want to transform fps from 100 to 30
    '''
    data_transformed = []
    for frameC, data in enumerate(data):
        if frameC%10==3 or frameC%10==7 or frameC%10==0 :
            data_transformed.append(data)
    
    return data_transformed


def getFingerArrangement_fromf0(frequency):
    '''
    music sample rate : 100 fps => 30fps
    output : a list of finger arrangement for each frame
    delay  : no
    '''
    
    print('=======GET FINGER ARRANGEMENTS=======')
    
    fingerData = frequency2fingerArrangement(frequency)
    
    fingerArr = []
    for frameC, arr in enumerate(fingerData):
        if frameC%10==3 or frameC%10==7 or frameC%10==0 :
            frameArr = []
            for index in arr.split(' ')[-1].split('-'):
                frameArr.append(int(index[0]))
            frameArr.append(arr.split(' ')[0])
            fingerArr.append(frameArr)
    
    print('Output Finger Arrangements : ' + str(len(fingerArr)) + ' frames')
    
    return fingerArr


from midiutil.MidiFile import MIDIFile
def writeNote2MIDIFile(fingerArr, musicloc):
    '''
    write the note extracted from the audio input via YAAPT into MIDI file
    
    note sequence: 30 fps
    set: 1 note = 1 beat
    bpm = 60*30 = 1800 beats/per minute
    
    '''
    
    print('=======WRITING NOTES INTO MIDI FILE=======')
    
    noteDictionary = { 'C' : 0,
                       'C#' : 1,
                       'D' : 2,
                       'D#' : 3,
                       'E' : 4,
                       'F' : 5,
                       'F#' : 6,
                       'G' : 7,
                       'G#' : 8,
                       'A' : 9,
                       'A#' : 10,
                       'B' : 11}
    
    note=[]
    for frame in fingerArr:
        note.append(frame[-1])
    
    note_number = []
    for note_frame in note:
        if note_frame == 'Stop':
            note_number.append(note_frame)
            continue
        if len(note_frame) == 2:
            note_name = note_frame[0]
        if len(note_frame) == 3:
            note_name = note_frame[0]+note_frame[2]
        note_number.append( (int(note_frame[1])+1) * 12 + noteDictionary[note_name] )
    
    current_note, counter = note_number[0], 0
    noteTypeLength = []
    for time, note_n in enumerate(note_number):
        if time == len(note_number)-1:
            noteTypeLength.append([current_note, counter+1])
        if note_n != current_note:
            noteTypeLength.append([current_note, counter])
            current_note, counter = note_n, 1
        else:
            counter += 1
            
    
    ## Write into MIDI file
    mf = MIDIFile(1)     # only 1 track
    track = 0   # the only track
    time = 0    # start at the beginning
    bpm = 30 * 60 # every note = 1 beat = 1/30 sec  # bpm = 60 sec = 60*30 = 1800 beats / per minute
    
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, bpm)
    
    channel = 0
    volume = 100
    
    # add note
    # mf.addNote(track, channel, pitch, time, duration, volume)
    time = 0
    for noteTL in noteTypeLength:
        if noteTL[0] == 'Stop':
            time += noteTL[1]
        else:
            mf.addNote(track, channel, noteTL[0], time, noteTL[1], volume)
            time += noteTL[1]
    
    filename = musicloc.split('/')[-1].split('.')[0] + '_MIDI'
    
    # write it to disk
    with open(filename+".mid", 'wb') as outf:
        mf.writeFile(outf)
        
    print('save in name of : ' + filename + '.mid' )
            
        
def getDownBeatViaMadmom(music_loc, beatsPerBar):
    '''
    beats per bar = a integar 
        ex.1 [3.4] and [6,8] => 3
        ex.2 [4,4] => 4
    output : a list of downbeat time position and which beat
    sample rate : 100hz => 30hz
    '''
    
    print('=======GET DOWNBEAT POSITION=======')
    print('Getting Downbeat position of ' + music_loc.split('/')[-1].split('.')[0] )
    
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=beatsPerBar, fps=100)
    act = RNNDownBeatProcessor()(music_loc)
    downbeat_raw = proc(act)
    
    downbeat = []
    for perbeat in downbeat_raw:
        downbeat.append( [ perbeat[1], int(perbeat[0]*30) ] )
    
    return downbeat


def fixCNNDelay(array, delay=32):
    '''
    just add delays of array[0]s to the front of the array in order to fix the delay caused by CNN
    '''
    
    for i in range(delay):
        array.insert(0, array[0])
    
    for i in range(delay-1):
        array.append(array[-1])
    
    return array


def rotate_around_point_highperf_Numpy(xy, radians, origin):
    """
    Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    
    adjust_xy = xy - origin
    
    rotate_matrix_X = np.array( (np.cos(radians), np.sin(radians)) )
    rotate_matrix_Y = np.array( (-np.sin(radians), np.cos(radians)) )
    
    rotate_xy = origin + np.array( (sum(adjust_xy * rotate_matrix_X), sum(adjust_xy * rotate_matrix_Y)) ) 

    return rotate_xy


def rotate_around_point_highperf(xy, radians, origin):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def list_of_Str_2_list_of_Float(input_list):
    output_list = []
    for i in input_list:
        output_list.append(float(i))
    return output_list


def get3DSampleData(sample_size, sampleCsvFoldLoc='./3DBowingTemplates/'):
    
    print('=======GETTING 3D TEMPLATE DATA=======')
    
    upBowSam, downBowSam = [], []
    for string in range(1,5):
        
        # up bow
        with open(sampleCsvFoldLoc + '3DBowingTemplate_' + str(string) + '_up.csv') as csvfile:
            rows = csv.reader(csvfile)
            upBowSam1 = []
            for isstart, row in enumerate(rows):
                if isstart == 0: 
                    continue
                frame = []
                for i, index in enumerate(row):
                    if i%4 == 3:
                        continue
                    frame.append(float(index))
                upBowSam1.append(frame)
            upBowSam.append(upBowSam1)
        
        # down bow
        with open(sampleCsvFoldLoc + '3DBowingTemplate_' + str(string) + '_down.csv') as csvfile:
            rows = csv.reader(csvfile)
            downBowSam1 = []
            for isstart, row in enumerate(rows):
                if isstart == 0: 
                    continue
                frame = []
                for i, index in enumerate(row):
                    if i%4 == 3:
                        continue
                    frame.append(float(index))
                downBowSam1.append(frame)
            downBowSam.append(downBowSam1)
    
    upBowSam_sameSize, downBowSam_sameSize = [], []
    # normalize to same size
    for sam in upBowSam:
        sam1 = []
        for i in range(sample_size):
            sam1.append( sam[ int( (len(sam)-1)* (i+1)/sample_size) ] )
        upBowSam_sameSize.append(sam1)
    
    for sam in downBowSam:
        sam1 = []
        for i in range(sample_size):
            sam1.append( sam[ int( (len(sam)-1)* (i+1)/sample_size) ] )
        downBowSam_sameSize.append(sam1)
        
    return upBowSam_sameSize, downBowSam_sameSize
                

def calculatingRightHandSkeleton_3D(bowlabel_pred, fingerArr, cutFrequency=15, sample_size=49):
    '''
    parameters :
        cutFrequency : smallest number of frame to finish a whole bow (if < cutFrequency => not to play a whole bow)
        
    maxspeed = a whole bowing in 15 frames(0.5s)
    '''
    
    print('=======CALCULATING RIGHT HAND=======')
    
    upBowSam, downBowSam = get3DSampleData(sample_size)

    ## bowing label 2 label type+length
    start_condition, counter = bowlabel_pred[0], 0
    labelTypeLength = []
    for frameC in range(len(bowlabel_pred)):
        if frameC == len(bowlabel_pred)-1:
            labelTypeLength.append([start_condition, counter+1])
        if bowlabel_pred[frameC] != start_condition:
            labelTypeLength.append([start_condition, counter])
            start_condition, counter = bowlabel_pred[frameC], 1
        else:
            counter += 1
    
    ## skeleton output base on SampleData(28*2)
    # situation initialize
    skeleton_output = []
    if labelTypeLength[0][0]=='up': 
        bowFramenow = 0
    else: 
        bowFramenow = sample_size
    
    cutFreqz = cutFrequency # avoid rapid change in 15/30=0.5 secs
    speedLimit = sample_size/cutFreqz
    
    for labelType in labelTypeLength:
        if labelType[0] == 'stop':
            for i in range(labelType[1]):
                skeleton_output.append(skeleton_output[len(skeleton_output)-1])
            continue
        
        if labelType[0] == 'up':
            lastFrame = sample_size - bowFramenow
        if labelType[0] == 'down':
            lastFrame = bowFramenow
            
        # if too fast => don't play a whole bow
        if lastFrame/labelType[1] >= speedLimit and labelType[1]*2 <= lastFrame+1:
            if labelType[0] == 'up':
                for i in range(labelType[1]):
                    skeleton_output.append(upBowSam[abs(fingerArr[len(skeleton_output)-2][0]-1)][bowFramenow])
                    bowFramenow += 2
            if labelType[0] == 'down':
                for i in range(labelType[1]):
                    skeleton_output.append(downBowSam[abs(fingerArr[len(skeleton_output)-2][0]-1)][sample_size-1-bowFramenow])
                    bowFramenow -= 2
                    
        else:
            if labelType[0] == 'up':
                for i in range(labelType[1]):
                    skeleton_output.append(upBowSam[abs(fingerArr[len(skeleton_output)-2][0]-1)][int(bowFramenow + lastFrame/labelType[1]*i)])
                bowFramenow = sample_size-1
            if labelType[0] == 'down':
                for i in range(labelType[1]):
                    skeleton_output.append(downBowSam[abs(fingerArr[len(skeleton_output)-2][0]-1)][sample_size-1-int(bowFramenow - lastFrame/labelType[1]*i)])
                bowFramenow = 0
        
    return skeleton_output
        
                
def getSampleData(sampleCsvLoc='./2DBowingTemplates/'):
    # 1,4 up; 2,3 down
    upBowSam, downBowSam = [], []
    for i in range(1,5):
        Data = []
        with open(sampleCsvLoc + 'BowingTemplate'+str(i) + '.csv') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                tmp=[]
                for c, index in enumerate(row):
                    if c%2 == 0:
                        tmp.append(float(index))
                    else:
                        tmp.append(-float(index))
                Data.append(tmp)
        
        Data_rev = []
        for frames in range(len(Data)-1, -1, -1):
            Data_rev.append(Data[frames])
        
        if i==1 or i==4:
            upBowSam.append(Data)
            downBowSam.append(Data_rev)
        if i==2 or i==3:
            downBowSam.append(Data)
            upBowSam.append(Data_rev)
    
    return upBowSam, downBowSam


def calculatingRightHandSkeleton(bowlabel_pred, fingerArr, cutFrequency=15):
    '''
    parameters :
        cutFrequency : smallest number of frame to finish a whole bow (if < cutFrequency => not to play a whole bow)
    '''
    
    print('=======CALCULATING RIGHT HAND=======')
    
    upBowSam, downBowSam = getSampleData()

    ## bowing label 2 label type+length
    start_condition, counter = bowlabel_pred[0], 0
    labelTypeLength = []
    for frameC in range(len(bowlabel_pred)):
        if frameC == len(bowlabel_pred)-1:
            labelTypeLength.append([start_condition, counter+1])
        if bowlabel_pred[frameC] != start_condition:
            labelTypeLength.append([start_condition, counter])
            start_condition, counter = bowlabel_pred[frameC], 1
        else:
            counter += 1
    
    ## skeleton output base on SampleData(28*2)
    # situation initialize
    skeleton_output = []
    if labelTypeLength[0][0]=='up': 
        bowFramenow = 0
    else: 
        bowFramenow = 27
    
    cutFreqz = cutFrequency # avoid rapid change in 15/30=0.5 secs
    speedLimit = 27/cutFreqz
    
    for labelType in labelTypeLength:
        if labelType[0] == 'stop':
            for i in range(labelType[1]):
                skeleton_output.append(skeleton_output[len(skeleton_output)-1])
            continue
        
        if labelType[0] == 'up':
            lastFrame = 27 - bowFramenow
        if labelType[0] == 'down':
            lastFrame = bowFramenow
            
        # if too fast => don't play a whole bow
        if lastFrame/labelType[1] >= speedLimit and labelType[1] <= lastFrame+1:
            if labelType[0] == 'up':
                for i in range(labelType[1]):
                    skeleton_output.append(upBowSam[abs(fingerArr[len(skeleton_output)-2][0]-1)][bowFramenow])
                    bowFramenow += 1
            if labelType[0] == 'down':
                for i in range(labelType[1]):
                    skeleton_output.append(downBowSam[abs(fingerArr[len(skeleton_output)-2][0]-1)][27-bowFramenow])
                    bowFramenow -= 1
        else:
            if labelType[0] == 'up':
                for i in range(labelType[1]):
                    skeleton_output.append(upBowSam[abs(fingerArr[len(skeleton_output)-2][0]-1)][int(bowFramenow + lastFrame / labelType[1] * i)])
                bowFramenow = 27
            if labelType[0] == 'down':
                for i in range(labelType[1]):
                    skeleton_output.append(downBowSam[abs(fingerArr[len(skeleton_output)-2][0]-1)][27-int(bowFramenow - lastFrame / labelType[1] * i)])
                bowFramenow = 0
        
    return skeleton_output


def skeleton2numpyType_3D(skeleton_list, joint_num=25, dimension=3):
    '''
    Original Input Data Format : (frames, joint_num*dimension)
    Transformed Output Data Format : (frames, joint, an array of (1,dimension) vector )
    '''
    
    skeleton_numpy = []
    for frameS in skeleton_list:
        frame_numpy = []
        for i in range(joint_num):
            position_vector = []
            for d in range(dimension):
                position_vector.append(frameS[i*dimension+d])
            frame_numpy.append(np.array(position_vector))
        skeleton_numpy.append(frame_numpy)
    
    return skeleton_numpy


def skeleton2numpyType(skeleton_list):
    '''
    Original Input Data Format : (frames, 10*2)
    Transformed Output Data Format : (frames, 10, an array of (1,2) vector )
    '''
    
    skeleton_numpy = []
    for frameS in skeleton_list:
        frame_numpy = []
        for i in range(10):
            position_vector = np.array( (frameS[i*2], frameS[i*2+1]) )
            frame_numpy.append(position_vector)
        skeleton_numpy.append(frame_numpy)
    
    return skeleton_numpy


def getPositionDis_on_3Dmodel(sample_size):
    '''
    3D model scaling
        human spine average = 70 cm (male:70~75cm, female:66~70cm)
        violin position distance (4/4 full size) : 1st/2nd/3rd/4th => 3.5/6.6/8.0/12.6 cm
    '''
    
    up, down = get3DSampleData(sample_size)
    merge = []
    for string in up:
        for frame in string:
            merge.append(frame)
    for string in down:
        for frame in string:
            merge.append(frame)
    
    spineBase, neck = 1, 4 
    spine_len_sum = 0
    for frame in merge:
        spine_len_sum += ( (frame[(spineBase-1)*3]-frame[(neck-1)*3])**2 +
                           (frame[(spineBase-1)*3+1]-frame[(neck-1)*3+1])**2 + 
                           (frame[(spineBase-1)*3+2]-frame[(neck-1)*3+2])**2 )**0.5
    spine_avg = spine_len_sum / len(merge)
    centimeter_scale = spine_avg / 70
    
    return [3.5*centimeter_scale, 6.6*centimeter_scale, 8.0*centimeter_scale, 12.6*centimeter_scale]
    

def addLeftHandPositionMovement_Numpy_3D(skeleton_output, fingerArr, sample_size=49):
    '''
    parameters :
       position_dis : the distance of moving one position on the violin cardboard 
    '''
    
    print('=======CALCULATING LEFT HAND=======')
    position_dis = getPositionDis_on_3Dmodel(sample_size)
    
    neck, hand_left, hand_tip_left, thumb_left = 4, 9, 10, 11
    for frameC, frameS in enumerate(skeleton_output):
        if frameC >= len(fingerArr):
            break
        unit_vec = ( frameS[neck-1] - frameS[hand_left-1] ) / np.linalg.norm(frameS[neck-1] - frameS[hand_left-1])
        frameS[hand_left-1] += position_dis[fingerArr[frameC][1]] * unit_vec
        frameS[hand_tip_left-1] += position_dis[fingerArr[frameC][1]] * unit_vec
        frameS[thumb_left-1] += position_dis[fingerArr[frameC][1]] * unit_vec
    
    return skeleton_output


def addLeftHandPositionMovement_Numpy(skeleton_output, fingerArr, position_dis=10):
    '''
    parameters :
       position_dis : the distance of moving one position on the violin cardboard 
    '''
    
    print('=======CALCULATING LEFT HAND=======')
    for frameC, frameS in enumerate(skeleton_output):
        unit_vec = ( frameS[0] - frameS[7] ) / np.linalg.norm(frameS[0]-frameS[7])
        frameS[7] += position_dis * unit_vec * (fingerArr[frameC][1]+fingerArr[frameC][2]*0)
    
    return skeleton_output


def space_rotation(xy, origin, phi, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    
    Parameters:
        xy, origin: numpy array
        phi, theta: radian
    """
    
    axis = np.array( (math.cos(3/2*math.pi-phi), 0, math.sin(3/2*math.pi-phi) ) )
    adjust_xy = xy - origin
    
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    rotated_adjust_xy = np.dot(rotation_matrix, adjust_xy)
    rotated_xy = rotated_adjust_xy + origin
    
    return rotated_xy


def addBodyMovement_Numpy_3D(skeleton_output, arousal, bodyDegree_max=60, body_tilt_direction_degree=150):
    '''
    parameters :
        bodyDegree_max : max extent of body tilt, unit of degree, default will be 60 degree
    '''
    
    print('=======CALCULATING BODY EMOTION MOVEMENT=======')
    
    hip_left, hip_right = 18, 22
    body_tilt_direction_radian = body_tilt_direction_degree * math.pi/180
    
    bodyRad_max = bodyDegree_max * np.pi/180
    for frameC, a in enumerate(arousal):
        if frameC >= len(skeleton_output):
            break
        bodyRad = -a * bodyRad_max
        offset = ( skeleton_output[frameC][hip_left-1] + skeleton_output[frameC][hip_right-1] ) / 2
        
        # only rotate upper body (number 1~17)
        for j, joint_pos_numpy in enumerate(skeleton_output[frameC]):
            if j >= 17:
                continue
            skeleton_output[frameC][j] = space_rotation(joint_pos_numpy, offset, body_tilt_direction_radian, bodyRad)
    
    return skeleton_output


def addBodyMovement_Numpy(skeleton_output, arousal, bodyDegree_max=60):
    '''
    parameters :
        bodyDegree_max : max extent of body tilt, unit of degree, default will be 60 degree
    '''
    
    print('=======CALCULATING BODY EMOTION MOVEMENT=======')
    
    bodyRad_max = bodyDegree_max * np.pi/180
    for frameC, a in enumerate(arousal):
        bodyRad = -a * bodyRad_max
        offset = ( skeleton_output[frameC][8] + skeleton_output[frameC][9] ) / 2
        
        for j, joint_pos_numpy in enumerate(skeleton_output[frameC]):
            qx, qy = rotate_around_point_highperf_Numpy(joint_pos_numpy, bodyRad, offset )
            q_xy = np.array([qx, qy])
            skeleton_output[frameC][j] = q_xy
    
    return skeleton_output


def addArousalMean2Downbeat(arousal, downbeat):
    ## arousal between downbeat
    for count in range(len(downbeat)):
        if count == 0:
            last_frame = 0
        else:
            last_frame = downbeat[count-1][1]
        
        mean = sum( arousal[ last_frame : downbeat[count][1] ] )/len(arousal[ last_frame : downbeat[count][1] ])
        downbeat[count].append(mean)

    return downbeat


def addDJShakyHeadMovement_3D(skeleton_output, arousal, downbeat,  songBeat, headDegree_max=20, body_tilt_direction_degree=150):
    '''
    parameters :
        bodyDegree_max : max extent of body tilt, unit of degree, default will be 20 degree
        songBeat : beat per bar
    '''
    
    print('=======CALCULATING HEAD EMOTION MOVEMENT=======')
    
    ## head rotation radians timelist
    headRad_max = headDegree_max * np.pi/180
    beat_per_bar = songBeat
    headRotation = []
    frame_now = 0
    for c, downbeat_ps in enumerate(downbeat):
    
        headRad = headRad_max * (downbeat_ps[2]+1)
        headRad_start = -headRad_max/4
        if c == 0 and downbeat_ps[0] != float(beat_per_bar) and downbeat_ps[0] != 1.0:
            headRotation.append(headRad_start)
            frame_now = 1
            
        if c == 0 and downbeat_ps[0] == float(beat_per_bar):
            for i in range(downbeat_ps[1]):
                headRotation.append(headRad_start)
                frame_now = downbeat_ps[1]
            continue
        
        if downbeat_ps[0] == 1.0:
            frames = downbeat_ps[1]-frame_now
            for i in range(frames):
                headRotation.append(headRad * (i+1) / frames + headRad_start )
            frame_now = downbeat_ps[1]
            
        if downbeat_ps[0] == float(beat_per_bar):
            frames = downbeat_ps[1]-frame_now
            for i in range(frames):
                headRotation.append( headRotation[-1] - ((downbeat[c-1][2]+1)*headRad_max-headRad_start)*1/frames )
            frame_now = downbeat_ps[1]
    
    body_tilt_direction_radian = body_tilt_direction_degree * math.pi/180
    
    ## head shake
    for frameC, headRad in enumerate(headRotation):
        
        headAndLeftHand = [3,4,5,6,7,8,9,10,11]
        spine_shoulder = 3
        offset = skeleton_output[frameC][spine_shoulder-1]
        for rotateP in headAndLeftHand:
            skeleton_output[frameC][rotateP-1] = space_rotation(skeleton_output[frameC][rotateP-1], offset, body_tilt_direction_radian, headRad)
    
    return skeleton_output


def addDJShakyHeadMovement(skeleton_output, arousal, downbeat,  songBeat, headDegree_max=20):
    '''
    parameters :
        bodyDegree_max : max extent of body tilt, unit of degree, default will be 20 degree
        songBeat : beat per bar
    '''
    
    print('=======CALCULATING HEAD EMOTION MOVEMENT=======')
    
    ## head rotation radians timelist
    headRad_max = headDegree_max * np.pi/180
    beat_per_bar = songBeat
    headRotation = []
    frame_now = 0
    for c, downbeat_ps in enumerate(downbeat):
    
        headRad = headRad_max * (downbeat_ps[2]+1)
        headRad_start = -headRad_max/4
        if c == 0 and downbeat_ps[0] != float(beat_per_bar) and downbeat_ps[0] != 1.0:
            headRotation.append(headRad_start)
            frame_now = 1
            
        if c == 0 and downbeat_ps[0] == float(beat_per_bar):
            for i in range(downbeat_ps[1]):
                headRotation.append(headRad_start)
                frame_now = downbeat_ps[1]
            continue
        
        if downbeat_ps[0] == 1.0:
            frames = downbeat_ps[1]-frame_now
            for i in range(frames):
                headRotation.append(headRad * (i+1) / frames + headRad_start )
            frame_now = downbeat_ps[1]
            
        if downbeat_ps[0] == float(beat_per_bar):
            frames = downbeat_ps[1]-frame_now
            for i in range(frames):
                headRotation.append( headRotation[-1] - ((downbeat[c-1][2]+1)*headRad_max-headRad_start)*1/frames )
            frame_now = downbeat_ps[1]
    
    ## head shake
    for frameC, headRad in enumerate(headRotation):
        
        offset = skeleton_output[frameC][1]
        for rotateP in [0,5,6,7]:
            qx, qy = rotate_around_point_highperf_Numpy(skeleton_output[frameC][rotateP], headRad, offset )
            q_xy = np.array([qx, qy])
            skeleton_output[frameC][rotateP] = q_xy
    
    return skeleton_output


def smoothSkeleton_Medfilt(skeleton_output, window_size, dimension, joints_num):
    '''
    smooth output skeleton with medfilt
    '''
    
    print('=======SMOOTH SKELETON VIA MEDFILT=======')
    
    for joints in range(joints_num):   
        for d in range(dimension):
            oneJointSke = []
            for frameC in range(len(skeleton_output)):
                oneJointSke.append(skeleton_output[frameC][joints][d])
            oneJointSke = medfilt(oneJointSke, window_size)
            for frameC in range(len(skeleton_output)):
                skeleton_output[frameC][joints][d] = oneJointSke[frameC]
    
    return skeleton_output


def skeleton2ListType(skeleton):
    skeleton_list = []
    for f in skeleton:
        tmp = []
        for j in f:
            tmp.append(j[0])
            tmp.append(j[1])
        skeleton_list.append(tmp)
    
    return skeleton_list
        
    
def getTimeMark(beethovenNo5Loc='./BeethovenNo5.csv'):
    timemark = []
    with open(beethovenNo5Loc, mode='r', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            timemark.append(row[-1])
    
    return timemark[1:]


def get_ItIsTimeToStop_Mark(beethovenNo5Loc='./BeethovenNo5.csv', stopMarkDuration=900):
    '''
    parameters:
        stopMarkDuration: frames
    '''
    itIsTimeToStopMark = []
    with open(beethovenNo5Loc, mode='r', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for count, row in enumerate(rows):
            if count == 0:
                continue
            if count >= stopMarkDuration:
                break
            magnify = []
            for index in row:
                if index.isnumeric():
                    index *= 100
                magnify.append(index)
            itIsTimeToStopMark.append(magnify)
    
    return itIsTimeToStopMark


def get_ItIsTimeToStop_Mark_2(beethovenNo5Loc='./BeethovenNo5.csv', stopMarkDuration=900):
    '''
    parameters:
        stopMarkDuration: frames
    '''
    itIsTimeToStopMark = []
    with open(beethovenNo5Loc, mode='r', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for count, row in enumerate(rows):
            if count == 1:
                for i in range(stopMarkDuration):  
                    itIsTimeToStopMark.append(row)
    
    return itIsTimeToStopMark
    
    
    
    
    
    
    
    
    
                
                
                
                
                
                