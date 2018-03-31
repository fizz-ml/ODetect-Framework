import argparse
from glob import glob
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from features.peakdetect import peakdetect
from scipy import interpolate

from scipy.signal import butter, lfilter
from scipy import signal

from features.simple_filter import SimpleButterFilter, stupid_local_norm, SimpleSplineFilter
from features.envelope import WindowEnvelopes, WindowEnvelopesAmplitude
from features.peak_feature import WindowPeakTroughPeriods, WindowPeakTroughPoints
from utils.thermistor import *

import pylab

"""
Visualizes the short time fourier transform of the signal.
Produces graphs for each signal in the specified dataset.
The graphs are composed of one plot of the ppg and co2 waveforms and one plot of the 2D STFT spectrogram.
Red points on the spectrogram correspond to the equivalent instantaneous frequency derived from the labels.
"""

def normalize(ppg_signal):
    ppg_signal = (ppg_signal-np.mean(ppg_signal, axis=0))/np.std(ppg_signal)
    return ppg_signal

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def visualize_dataset(dataset_path, plot):
    data = np.genfromtxt(dataset_path, delimiter=',')
    breath_signal = data[:,3].flatten()


    # Plot
    sample_freq = 200

    breath_signal = normalize(breath_signal)
    w=np.hanning(20)
    breath_avg = np.convolve(w/w.sum(),breath_signal,mode='same')

    # breath_butter_filter = SimpleButterFilter(sample_freq, 3/60, 40/60, order=1)
    # breath_filtered = breath_butter_filter.calc_feature(breath_signal)
    breath_butter_filter = SimpleSplineFilter(avg_win=60, ds=15, s=45.0)
    breath_filtered = stupid_local_norm(breath_butter_filter.calc_feature(breath_signal),10000)
    breath_filtered = normalize(breath_filtered)

    # tck = interpolate.splrep(np.arange(breath_filtered1.size)[::40], breath_filtered1[::40], s=25.0)
    # breath_filtered = interpolate.splev(np.arange(breath_filtered1.size), tck, der=0)

    # GT breath signal
    # breath_filtered = normalize(breath_filtered)
    ((breath_peak_idx, breath_peak_val, breath_peak_period),(breath_trough_idx, breath_trough_val, breath_trough_period)) = WindowPeakTroughPoints().calc_feature(breath_filtered, delta=1.0, lookahead=100)

    # ax2.plot(breath_trough_idx, np.reciprocal(breath_trough_period/sample_freq)*60, '+-', label="Thermistor Trough to Trough Frequency")
    fig, ax2 = plt.subplots(1,1)
    ax2.plot(breath_trough_idx, breath_trough_val, '.', markersize=20, label="Thermistor Trough to Trough Frequency")
    ax2.plot(breath_peak_idx, breath_peak_val, '.', markersize=20, label="Thermistor Trough to Trough Frequency")
    ax2.plot(breath_filtered, label="Filtered Thermistor")
    ax2.plot(np.arange(breath_signal.size)[::5], breath_signal[::5], '+', label="Raw Thermistor")
    plt.legend()
    plt.xlabel("Samples (at 200Hz)")
    plt.ylabel("RR in bpm")
    plt.title("Thermistor RR")
    plt.show()


    # Plot result
    fig, ax2 = plt.subplots(1,1)
    y = breath_filtered
    yp, yt = WindowEnvelopes().calc_feature(y, 300, 1.0, s=3)
    ax2.plot(y, label="Filtered Thermistor")
    # ax2.plot(yp, label="Max Envelope")
    # ax2.plot(yt, label="Min Envelope")
    # avg_env = (yt+yp)/2

    w = np.hanning(8000)
    mov_avg = np.convolve(w/w.sum(), y, 'same')
    avg_env = mov_avg
    ax2.plot(avg_env, label="Min Envelope")

    # w = np.hanning(8000)
    # mov_avg = np.convolve(w/w.sum(), y, 'same')

    ax2.plot(y, label="Filtered Thermistor")
    ax2.plot(np.arange(breath_avg.size)[::20], breath_avg[::20], '+', label="Raw Thermistor")
    plt.show()


    fig, ax2 = plt.subplots(1,1)
    ax2.plot(instant_bpm(breath_signal, sample_freq))
    ax2.plot(np.arange(breath_avg.size)[::20], breath_avg[::20], '+', label="Raw Thermistor")
    plt.show()
    # fig, ax2 = plt.subplots(1,1)
    # ax2.plot(np.gradient(breath_filtered), label="Gradient of Thermistor")
    # ax2.plot(breath_filtered, label="Filtered Thermistor")
    # plt.legend()
    # plt.show()



    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # plt.show()

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('csv_name', type=str, help='Name of the dataset under raw containing the data folder of h5 files to be processed.')
    parser.add_argument('thing', type=int, help='Name of the dataset under raw containing the data folder of h5 files to be processed.')
    args = parser.parse_args()

    input_path = args.csv_name
    thing = args.thing

    # Load some data
    # input_path = os.path.join('data', dataset_name, 'raw')
    print(visualize_dataset(input_path, thing))
