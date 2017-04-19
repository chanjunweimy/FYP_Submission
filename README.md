# Depression Detection from Speech
This is the repository containing of all my work for my Final Year Project, Depression Detection from Speech.

## Index
* [Abstract](README.md#abstract)
* [Acknowledgment](README.md#acknowledgment)
* [Repository Details](README.md#repository-details)
* [Audio Dataset](README.md#audio-dataset)
* [Tools Used](README.md#tools-used)
* [Author](README.md#author)

## Abstract
The lack of objective measures causes the most treatable metal illness, depression to be often under-diagnosed. 
Recent studies have shown that speech is a good indicator of depression, giving us a motivation to perform depression 
diagnosis using speech to create an objective measure. This project studies the use of state-of-the-art machine learning 
(ML) models including ensemble in predicting depression severity using audio features after optimizing the data. 
We obtain the audio data from Audio/Visual Emotion Challenge and Workshop 2016 (AVEC 2016) and aim to have a mean F1 of 0.8 
on the development (dev) set. Our work has successfully shown that AdaBoost (AB) trained using the mean of Zero-crossing Rate, 
Entropy of Energy, Spectral Spread, Spectral Entropy, Mel Frequency Cepstral Coefficients (MFCCs) and Chroma Deviation i
s a good model for depression prediction, which is able to predict Personal Health Questionnaire eight-item depression scale (PHQ-8) 
at mean F1 of 0.82 and Root Mean Square Error (RMSE) of 6.43. The results are better than other state-of-the-art models 
including the baselines at mean F1 of 0.5 and RMSE of 6.74. It also gives a mean F1 of 1 in multi-class classification, 
which predicts the depression level of individuals. In the future, we aim to further verify the model correctness and 
create an autonomous agent that could help the depressed patients.

## Acknowledgment
I wish to express my sincere thanks to my advisor, Professor Ooi Wei Tsang, for providing me with all the professional and
valuable guidance which is the key to my success. I am also grateful to Professor Bryan Kian Hsiang Low and 
Dr. Chua Tat-Seng from School of Computing for their guidance and sharing on their expertise.  
I take this opportunity to express gratitude to all faculty members of the Department for their help and support. 
I am extremely thankful to my high school Taylor App Competition (tête-à-tête) teammates, CS2108 group mates and CS4246 group mates for agreeing to let me extend the work on depression studies.
Also, I would like to thank the organiser of the Audio/Visual Emotion Challenge and Workshop 2016 (AVEC 2016) and 
the Audio/Visual Emotion Challenge and Workshop 2014 (AVEC 2014) for providing the depression corpus for us. 
Special thanks to the special ones, especially my family and friends, for their unceasing encouragement, 
support and attention. I also place on record, my gratitude to one and all, who directly or indirectly, 
have lent a hand in this venture.

## Repository Details
This repository contains 3 folders:
* code: contains all the python or batch script written to automate the process of generating the result.
* doc: contains all the documents such as the report and the presentation slides.
* data: the audio features extracted to be used to train the models after being prepared.

## Audio Dataset
We are using the [DAIC-WOZ database](http://dcapswoz.ict.usc.edu/) 
provided by AVEC2016 organizers that can be found in http://dcapswoz.ict.usc.edu/ 

## Tools Used
* [Python 2.7.12](https://www.python.org/downloads/release/python-2712/)
* [Numpy](https://github.com/numpy/numpy)
* [Scipy](https://github.com/scipy/scipy)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [GPy](https://github.com/SheffieldML/GPy)
* [GPyOpt](https://github.com/SheffieldML/GPyOpt)
* [Scikit-Feature](https://github.com/jundongl/scikit-feature)
* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
* [Windows whl files](http://www.lfd.uci.edu/~gohlke/pythonlibs/)
* [SoX](https://sourceforge.net/projects/sox/)
* [FFmpeg](https://ffmpeg.org/download.html)
* [Mendeley](https://www.mendeley.com/downloads)

## Author
[Chan Jun Wei](https://chanjunweimy.github.io/) is a NUS Computer Science Student specializing in Information Retrieval and 
Artificial Intelligence. You can find him in [LinkedIn](https://www.linkedin.com/in/junwei-chan-a07632a0/) too!
