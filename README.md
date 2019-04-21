# Low-Latency-Keyword-Spotting-System
NLP Project for CS6120 at Northeastern University

## Dataset download from [Tensorflow](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
(You will need to login to Google to be able to download it)

##### The structure of the project folders should be:
-Low-Latency-Keyword-Spotting-System
	-src
		-data_speech_commands_v0.02	(The extracted dataset folder)
		-input
		-weights
		-picts

After you have made this structure, begin by preprocessing the dataset.


## Models: 
1. Transfer Learning Using InceptionResnet
   - To run this model, first run *preprocess_inception.py* to convert the .wav files to .png spectrogram images.
   - Then run *inception-resnetv2.py*
2. Baseline CNN using MFCC Features
   - To run this model, first run *audio_feature_extraction.py* file to convert the .wav file to mfcc feature array saved in a numpy array
   - Then run *mfccModel.py*
3. Depth Separable CNN using MFCC Features
   - To run this model, first run *audio_feature_extraction.py* file to convert the .wav file to mfcc feature array saved in a numpy array
   - Then run *mfccModel_dscnn.py*
4. Baseline CNN using Logmel Filterbank Features
   - To run this model, first run *audio_feature_extraction_logmel.py* file to convert the .wav file to logmel filterbank feature array saved in a numpy array
   - Then run *logmelModel.py*
5. Depth Separable CNN using Logmel Filterbank Features
   - To run this model, first run *audio_feature_extraction_logmel.py* file to convert the .wav file to logmel filterbank feature array saved in a numpy array
   - Then run *logmelModel_dscnn.py*


