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
    ppg_signal = data[:,1].flatten()
    breath_signal = data[:,3].flatten()
    sample_freq = 200

    ppg_signal = normalize(ppg_signal)
    ppg_spline_filter = SimpleSplineFilter()
    ppg_filtered = stupid_local_norm(ppg_spline_filter.calc_feature(ppg_signal), 8000)

    breath_signal = normalize(breath_signal)
    breath_spline_filter = SimpleSplineFilter(avg_win=40, ds=40, s=15.0)
    breath_filtered = stupid_local_norm(breath_spline_filter.calc_feature(breath_signal), 4000)

    # tck = interpolate.splrep(np.arange(ppg_filtered1.size)[::40], ppg_filtered1[::40], s=25.0)
    # ppg_filtered = interpolate.splev(np.arange(ppg_filtered1.size), tck, der=0)

    # GT ppg signal
    # ppg_filtered = normalize(ppg_filtered)
    ((ppg_peak_idx, ppg_peak_val, ppg_peak_period),(ppg_trough_idx, ppg_trough_val, ppg_trough_period)) = WindowPeakTroughPoints().calc_feature(ppg_filtered, delta=0.2, lookahead=20)

    # ax2.plot(ppg_trough_idx, np.reciprocal(ppg_trough_period/sample_freq)*60, '+-', label="Thermistor Trough to Trough Frequency")

    # Plot
    fig, ax2 = plt.subplots(1,1)
    ax2.plot(ppg_trough_idx, ppg_trough_val, '.', markersize=10, label="PPG Troughs")
    ax2.plot(ppg_peak_idx, ppg_peak_val, '.', markersize=10, label="PPG Peaks")
    ax2.plot(ppg_filtered, label="Filtered PPG")
    # ax2.plot(np.arange(ppg_signal.size)[::2], ppg_signal[::2], '+', label="Raw Thermistor")

    plt.legend()
    plt.xlabel("Samples (at 200Hz)")
    plt.ylabel("RR in bpm")
    plt.title("Thermistor RR")
    plt.show()


    # Calculate gradient of PPG
    grad_feature = np.gradient(stupid_local_norm(ppg_filtered,1000))
    ((grad_peak_idx, grad_peak_val, grad_peak_period),(grad_trough_idx, grad_trough_val, grad_trough_period)) = WindowPeakTroughPoints().calc_feature(grad_feature, delta=0.2, lookahead=20)

    fig, ax2 = plt.subplots(1,1)
    ax2.plot(grad_feature, label="Gradient of Filtered PPG")
    ax2.plot(grad_trough_idx, grad_trough_val, '.', markersize=10, label="PPG Gradient Troughs")
    plt.show()

    # Interpolate period between troughs
    interp_ppg_peak_period, interp_ppg_trough_period = WindowPeakTroughPeriods().calc_feature(ppg_filtered, delta=0.1, lookahead=20)

    fig, ax2 = plt.subplots(1,1)
    ax2.plot(normalize(breath_filtered), label="Filtered Thermistor")
    ax2.plot(normalize(interp_ppg_trough_period), label="Filtered Thermistor")
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