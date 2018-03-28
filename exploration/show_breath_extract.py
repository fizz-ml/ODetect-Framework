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

"""
Visualizes the short time fourier transform of the signal.
Produces graphs for each signal in the specified dataset.
The graphs are composed of one plot of the ppg and co2 waveforms and one plot of the 2D STFT spectrogram.
Red points on the spectrogram correspond to the equivalent instantaneous frequency derived from the labels.
"""

def normalize(ppg_signal):
    ppg_signal = (ppg_signal-np.mean(ppg_signal, axis=0))/np.std(ppg_signal)
    return ppg_signal

def visualize_dataset(dataset_path, plot):
    data = np.genfromtxt(dataset_path, delimiter=',')
    breath_signal = data[:,3].flatten()

    # Plot
    fig, ax2 = plt.subplots(1,1)
    sample_freq = 200

    breath_signal = normalize(breath_signal)

    w=np.ones(20,'d')
    breath_filtered1 = np.convolve(w/w.sum(),breath_signal,mode='valid')

    breath_butter_filter = SimpleSplineFilter()
    breath_filtered = breath_butter_filter.calc_feature(breath_signal)

    # tck = interpolate.splrep(np.arange(breath_filtered1.size)[::40], breath_filtered1[::40], s=25.0)
    # breath_filtered = interpolate.splev(np.arange(breath_filtered1.size), tck, der=0)

    # GT breath signal
    # breath_filtered = normalize(breath_filtered)
    ((breath_peak_idx, breath_peak_val, breath_peak_period),(breath_trough_idx, breath_trough_val, breath_trough_period)) = WindowPeakTroughPoints().calc_feature(breath_filtered, delta=0.1, lookahead=200)

    # ax2.plot(breath_trough_idx, np.reciprocal(breath_trough_period/sample_freq)*60, '+-', label="Thermistor Trough to Trough Frequency")
    ax2.plot(breath_trough_idx, breath_trough_val, '+', label="Thermistor Trough to Trough Frequency")
    ax2.plot(breath_filtered, label="Filtered Thermistor")
    ax2.plot(np.arange(breath_signal.size)[::10], breath_signal[::10], '.', label="Raw Thermistor")
    plt.legend()
    plt.xlabel("Samples (at 200Hz)")
    plt.ylabel("RR in bpm")
    plt.title("Thermistor RR")
    plt.show()

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
