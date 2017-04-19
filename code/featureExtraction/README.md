# featureExtraction

This folder contains files provided by the pyAudioAnalysis library and some scripts used to automate the process of feature extraction.

## pyAudioAnalysis
An open-source python library named [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) that could perform features extraction is used to try out more audio features commonly used by others. As it is claimed that it has been used in depression classification, we decide to apply the features extracted by “pyAudioAnalysis” to the audio recordings from DAIC-WOZ. The files provided by this library are included to the folder, they are:
* [__init__.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/__init__.py)
* [analyzeMovieSound.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/analyzeMovieSound.py)
* [audioAnalysis.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/audioAnalysis.py)
* [audioAnalysisRecordAlsa.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/audioAnalysisRecordAlsa.py)
* [audioBasicIO.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/audioBasicIO.py)
* [audioFeatureExtraction.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/audioFeatureExtraction.py)
* [audioSegmentation.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/audioSegmentation.py)
* [audioTrainTest.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/audioTrainTest.py)
* [audioVisualization.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/audioVisualization.py)
* [convertToWav.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/convertToWav.py)
* [utilities.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/utilities.py)

## My work
These are the files that I have written for audio features extraction:
* [usingPyAudioFeature.bat](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/usingPyAudioFeature.bat): A Windows batch file written to automate the process of extracting audio features using pyAudioAnalysis. To use it, just double click it, or run "usingPyAudioFeature.bat" in command prompt. Feel free to change "dcapswoz_audio_participantonly_merged/dev/" and "dcapswoz_audio_participantonly_merged/train/" to correspond dev and train directory. In this batch file, audio features are extracted at a frame size of 50ms with 50% overlap.
* [averageTrainDev.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/featureExtraction/averageTrainDev.py): According to our assumption stating that depression severity remains constant over a certain period of time rather than changing at every moment in time, we assume that the speech signal should not vary a lot for depression and average of the features in different time frames can be taken. Thus, we take the mean of audio features from the feature file to obtain only one line of audio feature per file. To understand more about the variation of the audios, standard deviation of audio features are also taken. These are all automated in this python script.
