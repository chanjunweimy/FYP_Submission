# data
This folder contains the extracted audio features (denoted with ["X" files](README.md#x)) and the ground truth (denoted with ["Y" files](README.md#y)). As mentioned, the data is divided into:
1. dev set: the data that used to test the models.
2. train set: the data that used to train the models.

## X
These files are the extracted audio features for preprocessed audios. One row of audio features per audio file. They include:
* [devX.txt](https://github.com/chanjunweimy/FYP_Submission/blob/master/data/devX.txt): the extracted audio features for the audios from the development set. 
* [trainX.txt](https://github.com/chanjunweimy/FYP_Submission/blob/master/data/trainX.txt): the extracted audio features for the audios from the train set.

## Y
These files are the ground truth for the audios. They include ground truth:
* Binary Classification:
  * [devY_bin.txt](https://github.com/chanjunweimy/FYP_Submission/blob/master/data/devY_bin.txt): indicates whether the speaker of the audio in the dev set is depressed or not.
  * [trainY_bin.txt](https://github.com/chanjunweimy/FYP_Submission/blob/master/data/trainY_bin.txt): indicates whether the speaker of the audio in the train set is depressed or not.
* Regression:
  * [devY_sev.txt](https://github.com/chanjunweimy/FYP_Submission/blob/master/data/devY_sev.txt): consists of the phq8 score of the audio speakers in the dev set.
  * [trainY_sev.txt](https://github.com/chanjunweimy/FYP_Submission/blob/master/data/trainY_sev.txt): consists of the phq8 score of the audio speakers in the train set.
* Multi-class Classification:
  * [devY_multi.txt](https://github.com/chanjunweimy/FYP_Submission/blob/master/data/devY_multi.txt): records the depression level of the audio speakers in the dev set.
  * [trainY_multi.txt](https://github.com/chanjunweimy/FYP_Submission/blob/master/data/trainY_multi.txt): records the depression level of the audio speakers in the train set.
