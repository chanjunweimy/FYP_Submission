# code
This folder consists of 3 subfolders:
* [preprocess](https://github.com/chanjunweimy/FYP_Submission/tree/master/code/preprocess): contains all the codes used to preprocess the audios and data.
* [featureExtraction](https://github.com/chanjunweimy/FYP_Submission/tree/master/code/featureExtraction): contains all the codes used to extract the audio features from the audios.
* [model training](https://github.com/chanjunweimy/FYP_Submission/tree/master/code/model%20training): contains all the codes that used to train the model and generate the results.

## Programming Language
* Python
* Windows Batch Script

## Platform
Preferably **Windows**. I wrote the code in Windows. While most of the code are written in Python and can be run in any platforms, some of the codes are written as Windows Batch Script. Although they could be translated to python pretty easily and then all the codes can be run on any platform, they are not tried in any other platform, so I prefer you to run them on Windows.

## Setup
### Important Tools
There are several tools you **MUST** setup sequentially before proceeding:
* [SoX](https://sourceforge.net/projects/sox/): command line tool used to combine and clean the audios.
* [FFmpeg](https://ffmpeg.org/download.html): command line tool used to split the audios.
* [Python 2.7.12](https://www.python.org/downloads/release/python-2712/): the python version you would need to install.
* [Numpy](https://github.com/numpy/numpy): an important and basic python library that is used by many scientific python libraries and contains a lot of useful mathematical functions.
* [Scipy](https://github.com/scipy/scipy): an important and basic python library that is used by many scientific python libraries based on numpy and consists of many useful mathematical, scientific, engineering functions. 
* [Matplotlib](https://github.com/matplotlib/matplotlib): an important python library used to plot or visualize the data.
* [Scikit-learn](https://github.com/scikit-learn/scikit-learn): an important python module specialized for machine learning tasks.
* [Scikit-Feature](https://github.com/jundongl/scikit-feature)
* [Windows whl files](http://www.lfd.uci.edu/~gohlke/pythonlibs/): if for any reasons, you can't install some of the libraries mentioned above to your Windows machine, you can install using the Windows whl files provided in this link.
### Included libraries
On the other hand, I have already included this library in the [featureExtraction](https://github.com/chanjunweimy/FYP_Submission/tree/master/code/featureExtraction) folder (So you **NEED NOT** install them):
* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
### Libraries that are no longer used
These are some libraries that I have tried but no longer used (So you **NEED NOT** install them):
* [GPy](https://github.com/SheffieldML/GPy): a python Gaussian Process (GP) Framework. (Even though we still used the GP model provided by Scikit-learn)
* [GPyOpt](https://github.com/SheffieldML/GPyOpt): a python library used for GP optimization using GPy. We used it to perform Bayesian Optimization, but the code is not included here because it is not important.
