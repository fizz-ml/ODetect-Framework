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

from features.simple_filter import SimpleButterFilter
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

    breath_butter_filter = SimpleButterFilter(sample_freq, 3/60, 50/60, order=2)
    breath_filtered = breath_butter_filter.calc_feature(breath_signal)
    # breath_filtered = butter_bandpass_filter(breath_signal, 3/60, 30/60, sample_freq, order=2)

    # GT breath signal
    breath_filtered = normalize(breath_filtered)
    ((breath_peak_idx, breath_peak_val, breath_peak_period),(breath_trough_idx, breath_trough_val, breath_trough_period)) = WindowPeakTroughPoints().calc_feature(breath_filtered, delta=0.1, lookahead=200)
    # breath_trough_idx, breath_trough_val, breath_trough_period = calc_troughs(breath_filtered, delta=0.1, lookahead=200)
    # breath_peak_idx, breath_peak_val, breath_peak_period = calc_peaks(breath_filtered, delta=0.1, lookahead=200)

    ax2.plot(breath_trough_idx, np.reciprocal(breath_trough_period/sample_freq)*60, '+-', label="Thermistor Trough to Trough Frequency")
    # ax2.plot(breath_trough_idx, np.reciprocal(breath_trough_period/sample_freq)*60, '+-', label="Thermistor Trough to Trough Frequency")
    # ax2.plot(cnn_trough_idx*8, np.reciprocal(cnn_trough_period*8/sample_freq)*60, '+-', label="CNN Out Trough to Trough PFrequencyeriod")
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