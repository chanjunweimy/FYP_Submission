import numpy as np
import scipy.io.wavfile
import scipy.signal
from matplotlib import pylab
import matplotlib.pyplot as plt
from collections import Counter

AUDIO_DIR = "pyAudioAnalysisLib/preprocessed_audio/"
NON_DEPRESSED_SAMPLE = "303_merged.wav"
FIG_NON_DEPRESSED_SINE = "303_non_depressed_signal_wave.png"
FIG_NON_DEPRESSED_SPECTROGRAM = "303_non_depressed_signal_spectrogram.png"
FIG_DEPRESSED_SINE = "319_depressed_signal_wave.png"
FIG_DEPRESSED_SPECTROGRAM = "319_depressed_signal_spectrogram.png"
DEPRESSED_SAMPLE = "319_merged.wav"
PLOT_DURATION = 0.2
OVERLAP_CONSTANT = 2

def plot_f1_old(models_f1):
    ind = np.arange(len(models_f1))
    width = 0.42

    fig = plt.figure()
    ax = fig.add_subplot(111)

	
    plt.axhline(0.41, color='k', linestyle='solid', label="Baseline")    

    model_names, f1_depressed, f1_normal = zip(*models_f1)
    rects_normal = ax.bar(ind, f1_normal, width, color='b')
    rects_depressed = ax.bar(ind+width, f1_depressed, width, color='g')

    ax.set_ylabel('F1')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(model_names)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    ax.legend((rects_normal[0], rects_depressed[0]), ('normal', 'depresed'), loc=2)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%h,
                    ha='center', va='bottom')

    autolabel(rects_normal)
    autolabel(rects_depressed)

    plt.show()

def plot_mean_f1(models_f1):
	models_f1 = [i for i in models_f1 if i[1] > 0]

	ind = np.arange(len(models_f1))
	width = 0.42

	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	if len(models_f1) == 0:
		return
	
	model_names, f1_mean = zip(*models_f1)
	
	ax.set_ylabel('Mean F1')
	ax.set_xticks(ind)
	ax.set_xticklabels(model_names)
	labels = ax.get_xticklabels()
	plt.setp(labels, rotation=30, fontsize=10)

	def autolabel(rects):
		for rect in rects:
			h = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%h,
					ha='center', va='bottom')

	for i in ind:
		rects_mean = ax.bar(i, f1_mean[i], width)
		autolabel(rects_mean)

	plt.show()
	
	
def plot_f1(models_f1):
	models_f1 = [i for i in models_f1 if i[1] > 0]

	ind = np.arange(len(models_f1))
	width = 0.42

	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.axhline(0.50, color='k', linestyle='solid', label="Baseline")    

	if len(models_f1) == 0:
		return
	
	model_names, f1_mean, f1_depressed, f1_normal = zip(*models_f1)
	rects_mean = ax.bar(ind, f1_mean, width, color='b')
	rects_depressed = ax.bar(ind+width, f1_depressed, width, color='g')

	ax.set_ylabel('F1')
	ax.set_xticks(ind+width)
	ax.set_xticklabels(model_names)
	labels = ax.get_xticklabels()
	plt.setp(labels, rotation=30, fontsize=10)
	ax.legend((rects_mean[0], rects_depressed[0]), ('mean', 'depresed'), loc=2)

	def autolabel(rects):
		for rect in rects:
			h = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%h,
					ha='center', va='bottom')

	autolabel(rects_mean)
	autolabel(rects_depressed)

	plt.show()

def plot_bar(models_rmse):
    ind = np.arange(len(models_rmse))
    width = 0.42

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.axhline(6.7418, color='k', linestyle='solid', label="Baseline")    

    model_names, rmse_train, rmse_predict = zip(*models_rmse)
    rects_train = ax.bar(ind, rmse_train, width, color='b')
    rects_predict = ax.bar(ind+width, rmse_predict, width, color='g')

    ax.set_ylabel('RMSE')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(model_names)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    ax.legend((rects_train[0], rects_predict[0]), ('train', 'predict'), loc=2)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%h,
                    ha='center', va='bottom')

    autolabel(rects_train)
    autolabel(rects_predict)

    plt.show()

def plot_all_Y():
    with open("data/all/allY.txt", 'rb') as allY:
        cont = allY.readlines()
    cont = [ int(ea.strip()) for ea in cont]
    counter = Counter(cont)
    x = range(25)
    y = [ counter[k] for k in x]
    pylab.ylabel("Frequency")
    pylab.xlabel("Depression Severity [PHQ-8 Score]")
    plt.bar(x, y)
    plt.show()

def plot_Y(phq8, bin, figname, x_label, maxNum):
	if len(phq8) != len(bin):
		return
	depressed = np.zeros(maxNum)
	normal = np.zeros(maxNum)
	
	array_size = len(phq8)
	for i in range(array_size):
		if bin[i] == 0:
			normal[phq8[i]] = normal[phq8[i]] + 1
		elif bin[i] == 1:
			depressed[phq8[i]] = depressed[phq8[i]] + 1

	x = range(maxNum)

	pylab.ylabel("Frequency")
	pylab.xlabel(x_label)
	plt.bar(x, normal, label='Non-depressed', color='blue')
	plt.bar(x, depressed, label='Depressed', color='red', bottom = normal)
	plt.legend(loc='upper right')
	#plt.show()
	plt.savefig(figname, bbox_inches='tight')
	plt.clf()
	
def plot_Y_with_x(phq8, bin, figname, x_label, maxNum, x_tickslabels):
	if len(phq8) != len(bin):
		return
	depressed = np.zeros(maxNum)
	normal = np.zeros(maxNum)

	array_size = len(phq8)
	for i in range(array_size):
		if bin[i] == 0:
			normal[phq8[i]] = normal[phq8[i]] + 1
		elif bin[i] == 1:
			depressed[phq8[i]] = depressed[phq8[i]] + 1

	x = range(maxNum)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_ylabel("Frequency")
	ax.set_xlabel(x_label)
	ax.set_xticklabels(x_tickslabels)
	ax.set_xticks(x)
	ax.bar(x, normal, label='Non-depressed', color='blue')
	ax.bar(x, depressed, label='Depressed', color='red', bottom = normal)
	ax.legend(loc='upper right')
	plt.show()
	#plt.savefig(figname, bbox_inches='tight')
	#plt.clf()


def stats(fileName):
    with open(fileName, 'rb') as allY:
        cont = allY.readlines()
    cont = [ int(ea.strip()) for ea in cont]
    counter = Counter(cont)
    x = range(25)
    y = [ counter[k] for k in x]
    print("Std Dev:"+str(np.std(cont)))
    print("Mean:"+str(np.mean(cont)))
    print("Size:"+str(len(cont)))

#303 as non-depressed sample
#319 as depressed sample
def plotDepressedAndNormalSample():
	sampling_rate, depressed_signal = scipy.io.wavfile.read(AUDIO_DIR + DEPRESSED_SAMPLE)
	plotSineWave(depressed_signal, sampling_rate, PLOT_DURATION, FIG_DEPRESSED_SINE)
	#plotSpectrogram(depressed_signal, sampling_rate, PLOT_DURATION, FIG_DEPRESSED_SPECTROGRAM)
	
	sampling_rate, non_depressed_signal = scipy.io.wavfile.read(AUDIO_DIR + NON_DEPRESSED_SAMPLE)
	plotSineWave(non_depressed_signal, sampling_rate, PLOT_DURATION, FIG_NON_DEPRESSED_SINE)
	#plotSpectrogram(non_depressed_signal, sampling_rate, PLOT_DURATION, FIG_NON_DEPRESSED_SPECTROGRAM)


def plotSineWave(signal, sampling_rate, duration, figname):
	wave_size = int(sampling_rate * duration)
	time = np.arange(wave_size)
	signal = signal[0:wave_size]
	
	plt.plot(time, signal)
	plt.xlabel('time(s)')
	plt.ylabel('signal')
	plt.savefig(figname, bbox_inches='tight')
	plt.clf()
	#plt.show()

#function to plot spectrogram 
def plotSpectrogram(signal, sampling_rate, duration, figname):
	wave_size = int(sampling_rate * duration)
	blackman = scipy.signal.get_window('hamming', wave_size)
	
	f, t, Sxx = scipy.signal.spectrogram(signal, sampling_rate, window=blackman, 
				nperseg = wave_size, noverlap = wave_size / OVERLAP_CONSTANT, 
				nfft= wave_size, scaling = 'density')
	
	plt.pcolormesh(t, f, Sxx)
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()
	#plt.savefig(SPECTROGRAM_NAME, bbox_inches='tight')
