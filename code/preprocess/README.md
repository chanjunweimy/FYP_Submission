# preprocess
This folder contains all the codes that are used to preprocess the audios and some ground truth.

## Audio Preprocessing
The original audio contains many silence intervals caused by the equipment set up time for the interview. Besides, the original audio includes the voice of the interviewer, Ellie. There are also some noises in the environment that should be removed. Therefore, before extracting the audio features, the audio are pre-processed first to obtain a more accurate and reasonable result.

### Noise Reduction
Firstly, after obtaining the original audios from DAIC-WOZ corpus, we need to clean the audios because the original audios contain a lot of buzzing sounds. FFmpeg is used to extract a silence segment of the audio recording which contains noise. This type of segment is ideal for SoX to use in the noise reduction process, which can be easily found at the start or the end of the audio recordings. In fact, the first second of the audio recordings we obtained from the DAIC-WOZ is this type of segment, which is extracted by FFmpeg to be the noise sample. <br /><br />

Next, the noise sample is input into SoX to generate a noise profile, which is a representation of the audio recording that contains data of the consistent background noises, such as hiss or hum. While SoX is used to reduce such noises to produce a cleaner audio, the amount of noise removed is chosen to be 0.21. However, the noise reduction method provided by SoX is only moderately effective in removing consistent background noises, implying that more noises could be reduced if a better approach is available. Since most of the related studies that used the data provided by DAIC-WOZ did not perform noise removal, we can reasonably assume that the quality of the original audio is actually acceptable. Thus, the quality of the cleaned audio recordings after noise removal is deemed acceptable and a more effective noise reduction technique is not required. Files that are written to automate these processes are:

* [cleanaudio.bat](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/preprocess/cleanaudio.bat): a windows batch file. To use this file, place it into the folder that contains all the original audio files, then double click the batch file or run "cleanaudio.bat" in windows prompt.
* [cleanaudio.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/preprocess/cleanaudio.py): python version of the batch file. To use it, put it into the folder that contains all the original audio file and run "python cleanaudio.py" in command line.

### Obtaining Speech Segments
After reducing some noises from the original audio recordings, the cleaned recordings still consist of the speech segments of both the participants and the interviewer, Ellie. Since Ellie’s voice does not relate to the severity of the depression of the participants, her speech segments would add noise to the audio recordings. Fortunately, the transcripts of all audio recordings are provided in Comma-separated values (csv) files, which the speech segments of different speakers are provided with a time frame. Therefore, we could obtain the speech segments of the participants based on the time frames given in the transcripts. FFmpeg is used to split the audio segments, a python file:

* [extractUserSpeech.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/preprocess/extractUserSpeech.py): is written to automate the process of obtaining speech segments. To use it, place it into the folders that contain all the cleaned audios then run "python extractUserSpeech.py" in the command prompt.

### Combine Speech Segments
After extracting speech segments of the participants into different audio recordings, the speech segments of the same participant are then combined using the functions provided by SoX. The following file:

* [combineAudios.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/preprocess/combineAudios.py): is written to automate the process of combining the speech segments. To use it, put it into the folders that contains all the speech segments of different speakers, then run "python combineAudios.py" in the command prompt.

### Spliting the audios into train and dev folder
Using the csv files that specify which id is in train or dev set, get all the merged audios from a source directory, and split them into the train and dev folder in the target directory, a python script is written for this purpose:

* [splitTrainDev.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/preprocess/splitTrainDev.py): is written to automate the process of spliting the audios into the train and dev folder. Feel free to change the variables inside the file especially: "DIR_SOURCE" and "DIR_TARGET" as they are the relative file path of the source directory and target directory respectively.

### Convert PHQ-8 scores to Depression Level
As we propose to perform a multi-class classification which predicts the depression level of a speaker, we need to convert the PHQ-8 scores to depression level, we write a python script to automate this process:
* [convertToMultiClass.py](https://github.com/chanjunweimy/FYP_Submission/blob/master/code/preprocess/convertToMultiClass.py)
