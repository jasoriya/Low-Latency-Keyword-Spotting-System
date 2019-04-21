#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:53:46 2019

@author: ritikagupta
"""

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pandas as pd
import os
import numpy as np

testData = pd.read_csv('data_speech_commands_v0.02/testing_list.txt', sep="/", header=None)
testData.columns = ["label", "fileName"]  

validationData = pd.read_csv('data_speech_commands_v0.02/validation_list.txt', sep="/", header=None)
validationData.columns = ["label", "fileName"]  

temp= list()
for i in os.listdir("data_speech_commands_v0.02/"):
    if(os.path.isdir("data_speech_commands_v0.02/"+i)):
       for j in  os.listdir("data_speech_commands_v0.02/"+i):
           temp.append([i,j])
           
trainData=pd.DataFrame(temp,columns=["label", "fileName"])

testData['in_test']='yes'

new = trainData.merge(testData,on=['label','fileName'],how='left')
trainData=(new[new.in_test.isnull()])

trainData.drop(['in_test'],axis = 1, inplace = True)
testData.drop(['in_test'],axis = 1, inplace = True)

audio_path="data_speech_commands_v0.02"
      
pict_Path = './input/picts/train/'
test_pict_Path = './input/picts/test/'

if not os.path.exists(pict_Path):
    os.makedirs(pict_Path)

if not os.path.exists(test_pict_Path):
    os.makedirs(test_pict_Path)

subFolderList = []
for x in os.listdir(audio_path):
    if os.path.isdir(audio_path + '/' + x):
        subFolderList.append(x)
        if not os.path.exists(pict_Path + '/' + x):
            os.makedirs(pict_Path +'/'+ x)
        if not os.path.exists(test_pict_Path + '/' + x):
            os.makedirs(test_pict_Path +'/'+ x)
            

train_audio_path = (audio_path  + "/"+trainData['label']+"/" + trainData['fileName'])
test_audio_path = (audio_path  + "/"+testData['label']+"/" + testData['fileName'])


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def wav2img(wav_path, targetdir='', figsize=(4,4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """

    #fig = plt.figure(figsize=figsize)    
    # use soundfile library to read in the wave files
    samplerate, test_sound  = wavfile.read(wav_path)
    _, spectrogram = log_specgram(test_sound, samplerate)
    
    ## create output path
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir +'/'+ output_file
    #plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.imsave('%s.png' % output_file, spectrogram)
    plt.close()


# WAVEFORM
#def wav2img_waveform(wav_path, targetdir='', figsize=(4,4)):
#    samplerate,test_sound  = wavfile.read(sample_audio[0])
#    fig = plt.figure(figsize=figsize)
#    plt.plot(test_sound)
#    plt.axis('off')
#    output_file = wav_path.split('/')[-1].split('.wav')[0]
#    output_file = targetdir +'/'+ output_file
#    plt.savefig('%s.png' % output_file)
#    plt.close()

        
for i, x in enumerate(subFolderList):
    print(i, ':', x)
    temp=trainData.loc[trainData['label']==x]
    all_files=(audio_path  + "/"+temp['label']+"/" + temp['fileName'])
    for file in all_files:
        wav2img(file, pict_Path + x)



for i, x in enumerate(subFolderList):
    print(i, ':', x)
    temp=testData.loc[testData['label']==x]
    all_files=(audio_path  + "/"+temp['label']+"/" + temp['fileName'])
    for file in all_files:
        wav2img(file, test_pict_Path + x)
        

labelList = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_background_noise_']
trainData['new_labels'] = trainData['label'].apply(lambda x: 'unknown' if x not in labelList else x)
testData['new_labels'] = testData['label'].apply(lambda x: 'unknown' if x not in labelList else x)

trainData['fileName'] = trainData.apply(lambda x: x['label'] + '/'+ x['fileName'], axis=1)
testData['fileName'] = testData.apply(lambda x: x['label'] + '/'+ x['fileName'], axis=1)


labelsTrain = pd.concat([trainData,pd.get_dummies(trainData['new_labels'])],axis=1)
labelsTrain.drop(['label', 'new_labels'],axis = 1, inplace = True)
labelsTrain['fileName'] = labelsTrain['fileName'].apply(lambda x: x.replace('.wav', '.png', 1))
labelsTrain.to_csv('labelsTrain.csv', index=False)

labelsTest = pd.concat([testData,pd.get_dummies(testData['new_labels'])],axis=1)
labelsTest.drop(['label', 'new_labels'],axis = 1, inplace = True)
labelsTest['fileName'] = labelsTest['fileName'].apply(lambda x: x.replace('.wav', '.png', 1))
labelsTest.insert(1, '_background_noise_', 0)
labelsTest.to_csv('labelsTest.csv', index=False)