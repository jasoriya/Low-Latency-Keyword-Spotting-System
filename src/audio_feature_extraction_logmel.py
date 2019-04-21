# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:26:29 2019

@author: shreyans
"""

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pandas as pd
import os
import numpy as np
import librosa
import librosa.display
from python_speech_features import logfbank

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

def dict_to_array(mydict, extract):
    valList = []
    if extract =='k':
        for value in mydict.keys():
            valList.append(value)
    elif extract=='v':
        for value in mydict.values():
            valList.append(value)
    return np.array(valList)

def wav2logmel(file, max_pad_len=32):
    wave, sr = librosa.load(file, mono=True, sr=None)
    wave = wave[::3]
    fbank_feat = logfbank(wave, 16000)
    pad_width = max_pad_len - fbank_feat.shape[0]
    if pad_width < 0:
        fbank_feat = fbank_feat[:max_pad_len]
    else:
        fbank_feat = np.pad(fbank_feat, pad_width=((0, pad_width), (0, 0)), mode='constant')
    return fbank_feat
    

logmelTrain = dict()

for i, x in enumerate(subFolderList):
    print(i, ':', x)
    temp=trainData.loc[trainData['label']==x]
    all_files=(audio_path  + "/"+temp['label']+"/" + temp['fileName'])
    for file in all_files:
        logmelF = wav2logmel(file, max_pad_len=32)
        logmelTrain[file] = logmelF

logmelTest = dict()

for i, x in enumerate(subFolderList):
    print(i, ':', x)
    temp=testData.loc[testData['label']==x]
    all_files=(audio_path  + "/"+temp['label']+"/" + temp['fileName'])
    for file in all_files:
        mfccF = wav2logmel(file, max_pad_len=32)
        logmelTest[file] = mfccF
    

#Train Labels
labelList = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_background_noise_']
trainData['new_labels'] = trainData['label'].apply(lambda x: 'unknown' if x not in labelList else x)
testData['new_labels'] = testData['label'].apply(lambda x: 'unknown' if x not in labelList else x)

trainData['fileName'] = trainData.apply(lambda x: x['label'] + '/'+ x['fileName'], axis=1)
testData['fileName'] = testData.apply(lambda x: x['label'] + '/'+ x['fileName'], axis=1)


labelsTrain = pd.concat([trainData,pd.get_dummies(trainData['new_labels'])],axis=1)
labelsTrain.drop(['label', 'new_labels'],axis = 1, inplace = True)
labelsTrain.sort_values(['fileName'], inplace=True)
labelsTrain.reset_index(drop=True, inplace=True)

logmelTrainDF = pd.DataFrame(list(logmelTrain.items()), columns=['fileName', 'values'])
logmelTrainDF.sort_values(['fileName'], inplace=True)
logmelTrainDF.reset_index(drop=True, inplace=True)
labelsTrain['fileName'] = logmelTrainDF['fileName']
labelsTrain.to_csv('input/logmel_labelsTrain.csv', index=False)
np.save("input/logmelTrainFeatures.npy", np.array(logmelTrainDF['values'].tolist()))

#Test Labels

labelsTest = pd.concat([testData,pd.get_dummies(testData['new_labels'])],axis=1)
labelsTest.drop(['label', 'new_labels'],axis = 1, inplace = True)
labelsTest.insert(1, '_background_noise_', 0)
labelsTest.sort_values(['fileName'], inplace=True)
labelsTest.reset_index(drop=True, inplace=True)

logmelTestDF = pd.DataFrame(list(logmelTest.items()), columns=['fileName', 'values'])
logmelTestDF.sort_values(['fileName'], inplace=True)
logmelTestDF.reset_index(drop=True, inplace=True)
labelsTest['fileName'] = logmelTestDF['fileName']
labelsTest.to_csv('input/logmel_labelsTest.csv', index=False)
np.save("input/logmelTestFeatures.npy", np.array(logmelTestDF['values'].tolist()))
