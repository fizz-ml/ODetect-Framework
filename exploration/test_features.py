import argparse
from glob import glob
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate

from scipy.signal import butter, lfilter
from scipy import signal

from features.feature import IdentityWindowFeature
from features.simple_filter import SimpleSplineFilter, SimpleButterFilter, SimpleLocalNorm, normalize
from features.envelope import WindowEnvelopes, WindowEnvelopesAmplitude
from features.peak_feature import WindowPeakTroughPeriods, WindowPeakTroughPoints

from utils import thermistor

"""
"""

def plot_max():
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

def visualize_dataset(dataset_path, plot):
    data = h5py.File(dataset_path, 'r')['data']

    input_signal = data['signal']
    target_signal = data['target']
    sampling_rate = data.attrs['sampling_rate']
    ts = np.linspace(0, input_signal.size/sampling_rate, input_signal.size)


    # Plot the raw signal
    fig, ax = plt.subplots(2,1)
    ax[0].set_title("Raw Input Signal")
    ax[0].plot(ts, input_signal, label="Raw Input Signal")

    ax[1].set_title("Raw Target Signal")
    ax[1].plot(ts, target_signal, label="Raw Target Signal")

    ax[0].set_xlabel("Time in seconds")
    ax[1].set_xlabel("Time in seconds")
    plot_max()


    # SimpleLocalNorm
    input_signal = normalize(input_signal)
    local_input_signal = SimpleLocalNorm(sampling_rate, [], {'local_window_length': 20}).calc_feature(input_signal)

    fig, ax = plt.subplots(2,1)
    ax[0].set_title("Norm Input Signal")
    ax[0].plot(ts, input_signal, label="Norm Input Signal")

    ax[1].set_title("SimpleLocalNorm of Norm Input Signal")
    ax[1].plot(ts, local_input_signal, label="SimpleLocalNorm")

    ax[0].set_xlabel("Time in seconds")
    ax[1].set_xlabel("Time in seconds")
    plot_max()


    # SimpleButterFilter
    filtered_input = SimpleButterFilter(sampling_rate, [], {'low_cut': 3/60, 'high_cut': 90/60, 'order': 3}).calc_feature(input_signal)

    fig, ax = plt.subplots(2,1)
    ax[0].set_title("Norm Input Signal")
    ax[0].plot(ts, input_signal, label="Norm Input Signal")

    ax[1].set_title("SimpleButterFilter of Norm Input Signal")
    ax[1].plot(ts, filtered_input, label="SimpleButterFilter")

    ax[0].set_xlim(0,100)
    ax[1].set_xlim(0,100)

    ax[0].set_xlabel("Time in seconds")
    ax[1].set_xlabel("Time in seconds")
    plot_max()


    # SimpleSplineFilter
    norm_target = SimpleLocalNorm(sampling_rate, [], {'local_window_length': 80}).calc_feature(input_signal)
    norm_target = normalize(norm_target)
    filtered_target = SimpleSplineFilter(sampling_rate, [], {'local_window_length': 30/200, 'ds': 20, 's': 35.0}).calc_feature(norm_target)
    alt_filtered_target = SimpleButterFilter(sampling_rate, [], {'low_cut': 3/60, 'high_cut': 40/60, 'order': 2}).calc_feature(norm_target)

    fig, ax = plt.subplots(1,1)
    ax.set_title("SimpleSplineFilter on Input Signal")
    ax.plot(ts, norm_target, '+', label="Norm Input Signal")
    ax.plot(ts, filtered_target, label="SimpleSplineFilter")
    ax.plot(ts, alt_filtered_target, label="SimpleButterFilter")

    ax.set_xlim(0,200)

    ax.set_xlabel("Time in seconds")

    plt.legend()
    plot_max()


    # WindowPeakTroughPeriods
    pisp = WindowPeakTroughPeriods(sampling_rate, [], {'lookahead_length': 5/200, 'delta': 0.02, 'interp': 'spline', 's': 0, "toggle_p_t": True}).calc_feature(filtered_input)
    pist = WindowPeakTroughPeriods(sampling_rate, [], {'lookahead_length': 5/200, 'delta': 0.02, 'interp': 'spline', 's': 0, "toggle_p_t": False}).calc_feature(filtered_input)

    fig, ax = plt.subplots(1,1)
    ax.set_title("SimpleSplineFilter on Target Signal")
    ax.plot(ts, normalize(pisp), label="PISP")
    ax.plot(ts, normalize(pist), label="PIST")
    ax.plot(ts, filtered_target, label="Filtered Target")

    ax.set_xlim(0,400)
    ax.set_ylim(-5,5)

    ax.set_xlabel("Time in seconds")

    plt.legend()
    plot_max()


    # WindowEnvelopesAmplitude
    ase = WindowEnvelopesAmplitude(sampling_rate, [], {'lookahead_length': 5/200, 'delta': 0.02}).calc_feature(filtered_input)

    fig, ax = plt.subplots(1,1)
    ax.set_title("Amplitude by Spline Envelopes on Filtered Input")
    ax.plot(ts, normalize(ase), label="Amplitude by Spline Envelopes")
    ax.plot(ts, filtered_target, label="Filtered Target")

    ax.set_xlim(0,400)
    ax.set_ylim(-5,5)

    ax.set_xlabel("Time in seconds")

    plt.legend()
    plot_max()

    # Instant BPM
    bpm = thermistor.instant_bpm(target_signal, sampling_rate)
    fig, ax = plt.subplots(1,1)
    ax.plot(ts, bpm, label="Filtered Target")
    plot_max()


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('csv_name', type=str, help='Name of the dataset under raw containing the data folder of h5 files to be processed.')
    parser.add_argument('thing', type=int, help='Name of the dataset under raw containing the data folder of h5 files to be processed.')
    args = parser.parse_args()

    input_path = args.csv_name
    thing = args.thing

    print(visualize_dataset(input_path, thing))
