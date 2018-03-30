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
from features.peak_feature import WindowPeakTroughPeriods

"""
Visualizes the short time fourier transform of the signal.
Produces graphs for each signal in the specified dataset.
The graphs are composed of one plot of the ppg and co2 waveforms and one plot of the 2D STFT spectrogram.
Red points on the spectrogram correspond to the equivalent instantaneous frequency derived from the labels.
"""

def calc_troughs(co2signal, lookahead=5, delta=0.02):
    '''
        Returns the index of troughs and period between them for the supplied co2 signal.
        Note: Troughs are used over peaks since the troughs are more sharp for the co2 signal.
    '''
    # Only grab the troughs not the peaks
    _, troughs_idx_value = peakdetect(co2signal, lookahead=lookahead, delta=delta)
    # Just need the position
    troughs_idx = np.asarray([x[0] for x in troughs_idx_value])
    troughs_value = np.asarray([x[1] for x in troughs_idx_value])
    # Also compute the period between the previous and the current trough
    troughs_period = np.empty_like(troughs_idx)
    # For the very first trough assume same period as next one
    troughs_period[1:] = np.diff(troughs_idx)
    troughs_period[0] = troughs_period[1]
    return [troughs_idx, troughs_value, troughs_period]

def calc_peaks(co2signal, lookahead=5, delta=0.02):
    '''
        Returns the index of troughs and period between them for the supplied co2 signal.
        Note: Troughs are used over peaks since the troughs are more sharp for the co2 signal.
    '''
    # Only grab the troughs not the peaks
    troughs_idx_value, _ = peakdetect(co2signal, lookahead=lookahead, delta=delta)
    # Just need the position
    troughs_idx = np.asarray([x[0] for x in troughs_idx_value])
    troughs_value = np.asarray([x[1] for x in troughs_idx_value])
    # Also compute the period between the previous and the current trough
    troughs_period = np.empty_like(troughs_idx)
    # For the very first trough assume same period as next one
    troughs_period[1:] = np.diff(troughs_idx)
    troughs_period[0] = troughs_period[1]
    return [troughs_idx, troughs_value, troughs_period]

def interpolate_spline(in_idx, in_val, out_idx, s=0):
    tck = interpolate.splrep(in_idx, in_val, s=s)
    return interpolate.splev(out_idx, tck, der=0)

def normalize(ppg_signal):
    ppg_signal = (ppg_signal-np.mean(ppg_signal, axis=0))/np.std(ppg_signal)
    return ppg_signal

def stupid_local_norm(sig, window_size=2000):
    win = signal.hann(window_size)
    sig_mean = signal.convolve(sig, win, mode='same') / sum(win)
    shift_sig = sig - sig_mean

    abs_sig = np.abs(shift_sig)
    win = signal.hann(window_size)
    sig_std = signal.convolve(abs_sig, win, mode='same') / sum(win)

    norm_sig = shift_sig/sig_std

    return norm_sig


def visualize_dataset(dataset_path, plot):
    data = np.genfromtxt(dataset_path, delimiter=',')
    ppg_signal = data[:,1].flatten()
    breath_signal = data[:,3].flatten()
    print(ppg_signal.shape)

    # Plot
    fig, ax2 = plt.subplots(1,1)
    sample_freq = 200

    ppg_signal = normalize(ppg_signal)
    breath_signal = normalize(breath_signal)

    if plot == -2:
        # norm_sig = stupid_local_norm(ppg_filtered)
        # fig, [ax1,ax2] = plt.subplots(1,2)
        # ax1.plot(ppg_signal, "Raw PPG")
        ax2.set_title("Raw PPG")
        ax2.plot(ppg_signal, label="Raw PPG")
        ax2.plot(breath_signal, label="Raw Breath")
        plt.legend()
        # plt.plot(norm_sig)
        plt.show()

    # Filter the signals
    # ax2.plot(breath_signal)
    ppg_butter_filter = SimpleButterFilter(sample_freq, 3/60, 90/60, order=3)
    ppg_filtered = ppg_butter_filter.calc_feature(ppg_signal)
    # ppg_filtered = butter_bandpass_filter(ppg_signal, 3/60, 90/60, sample_freq, order=3)

    print(ppg_filtered.shape)
    if plot == -1:
        norm_sig = stupid_local_norm(ppg_filtered)
        fig, [ax1,ax2] = plt.subplots(1,2)
        ax1.plot(ppg_filtered)
        ax1.set_title("Filtered PPG")
        ax2.plot(norm_sig)
        ax2.set_title("Normalized Filtered PPG")
        # plt.plot(norm_sig)
        plt.show()

    ppg_filtered = stupid_local_norm(ppg_filtered)
    print(ppg_filtered.shape)

    breath_butter_filter = SimpleButterFilter(sample_freq, 3/60, 50/60, order=5)
    breath_filtered = breath_butter_filter.calc_feature(breath_signal)
    # breath_filtered = butter_bandpass_filter(breath_signal, 3/60, 30/60, sample_freq, order=2)

    # Calc the peaks
    ppg_trough_idx, ppg_trough_val, ppg_trough_period = calc_troughs(ppg_filtered)
    ppg_peak_idx, ppg_peak_val, ppg_peak_period = calc_peaks(ppg_filtered)


    if plot == 1:
        ax2.plot(breath_filtered, label="Filtered PPG")
        ax2.plot(ppg_filtered, label="Filtered Thermistor")
        plt.legend()
        plt.xlabel("Samples")
        plt.title("Filtered PPG and Thermistor")
        plt.show()

    if plot == 2:
        ax2.plot(ppg_filtered)
        ax2.scatter(ppg_trough_idx, ppg_trough_val)
        ax2.scatter(ppg_peak_idx, ppg_peak_val)

        plt.xlabel("Samples")
        plt.title("Peak detector on filtered ppg")
        plt.show()

    # Calc the envelopes
    # ppg_trough_envelope = interpolate_spline(ppg_trough_idx, ppg_trough_val, np.arange(ppg_filtered.size))
    # ppg_peak_envelope = interpolate_spline(ppg_peak_idx, ppg_peak_val, np.arange(ppg_filtered.size))
    ppg_envelopes_feature = WindowEnvelopes()
    ppg_peak_envelope, ppg_trough_envelope = ppg_envelopes_feature.calc_feature(ppg_filtered)
    ppg_amplitude_feature = WindowEnvelopesAmplitude()
    ppg_envelope_amplitude = ppg_amplitude_feature.calc_feature(ppg_filtered)

    if plot == 0:
        ax2.plot(ppg_signal, label="Raw PPG")
        ax2.plot(ppg_peak_envelope)
        ax2.plot(ppg_trough_envelope)
        ax2.plot(ppg_filtered, label="Filtered PPG")
        plt.legend()
        plt.xlabel("Samples")
        plt.title("Filtered PPG and Thermistor")
        plt.show()

    if plot == 3:
        ax2.plot(ppg_filtered)
        ax2.plot(ppg_trough_envelope)
        ax2.plot(ppg_peak_envelope)
        # ax2.scatter(butter_bandpass_filter(ppg_signal, 3/60, 60/60, sample_freq, order=3))
        plt.xlabel("Samples")
        plt.title("Spline interpolation to get envelope of ppg")
        plt.show()

    if plot == 4:
        ax2.plot(normalize(breath_filtered), label="Filtered Thermistor")
        ax2.plot(3*normalize(ppg_envelope_amplitude), label="Amplitude between envelopes")
        plt.legend()
        plt.xlabel("Samples")
        plt.title("Calculate signal amplitude by subtracting envelopes")
        plt.show()

    """
    interp_ppg_peak_period = np.interp(np.arange(ppg_filtered.size), ppg_peak_idx, normalize(ppg_peak_period))
    interp_ppg_trough_period = np.interp(np.arange(ppg_filtered.size), ppg_trough_idx, normalize(ppg_trough_period))
    """

    # interp_ppg_peak_period = interpolate_spline(ppg_peak_idx, normalize(ppg_peak_period), np.arange(ppg_filtered.size))
    # interp_ppg_trough_period = interpolate_spline(ppg_trough_idx, normalize(ppg_trough_period), np.arange(ppg_filtered.size))
    interp_ppg_peak_period, interp_ppg_trough_period = WindowPeakTroughPeriods().calc_feature(ppg_filtered)

    if plot == 5:
        ax2.plot(normalize(breath_filtered), label="Filtered Thermistor")
        # ax2.scatter(ppg_peak_idx, normalize(ppg_peak_period), label="Period between peaks")
        # ax2.scatter(ppg_trough_idx, normalize(ppg_trough_period), label="Period between troughs")
        ax2.plot(normalize(interp_ppg_peak_period), label="Smoothed period between peaks")
        # ax2.plot(interp_ppg_trough_period)
        plt.legend()
        plt.xlabel("Samples")
        plt.title("Period between peaks in filtered ppg")
        plt.show()

    if plot == 6:
        # GT breath signal
        breath_filtered = normalize(breath_filtered)
        breath_trough_idx, breath_trough_val, breath_trough_period = calc_troughs(breath_filtered, delta=0.1, lookahead=200)
        breath_peak_idx, breath_peak_val, breath_peak_period = calc_peaks(breath_filtered, delta=0.1, lookahead=200)

        """
        # ppg_t_p_trough_idx, ppg_t_p_trough_val, ppg_t_p_trough_period = calc_troughs(interp_ppg_trough_period)
        ax2.plot((breath_filtered), label="Filtered Thermistor")
        ax2.plot((breath_signal), label="Unfiltered Thermistor")
        # ax2.plot(np.reciprocal(interp_breath_peak_period/sample_freq)*60, label="Thermistor Peak to Peak Period")

        ax2.scatter(breath_trough_idx, breath_trough_val, label="Thermistor Trough ")
        # ax2.scatter(breath_trough_idx, np.reciprocal(breath_trough_period/sample_freq)*60, label="Thermistor Trough to Trough Period", marker="+")
        """

        # ax2.scatter(ppg_t_p_trough_idx, np.reciprocal(ppg_t_p_trough_period/sample_freq)*60, label="PIST Feature Trough to Trough Period", marker="+")

        from utils.breath_cnn import main
        cnn_out = (SimpleButterFilter(sample_freq, 3/60, 90/60, order=3).calc_feature(main()))
        cnn_trough_idx, cnn_trough_val, cnn_trough_period = calc_troughs(cnn_out)
        cnn_peak_idx, cnn_peak_val, cnn_peak_period = calc_troughs(cnn_out)
        """
        ax2.plot(np.arange(cnn_out.size)*8, cnn_out)
        ax2.scatter(cnn_trough_idx*8, cnn_trough_val, label="Thermistor Trough ")
        ax2.plot((breath_filtered), label="Filtered Thermistor")
        ax2.scatter(breath_trough_idx, breath_trough_val, label="Thermistor Trough ")
        """


        # ax2.scatter(cnn_trough_idx*8, np.reciprocal(cnn_trough_period*8/sample_freq)*60, label="CNN Out Trough to Trough Period", marker="+")
        ax2.plot(breath_trough_idx, np.reciprocal(breath_trough_period/sample_freq)*60, '+-', label="Thermistor Trough to Trough Frequency")
        ax2.plot(cnn_trough_idx*8, np.reciprocal(cnn_trough_period*8/sample_freq)*60, '+-', label="CNN Out Trough to Trough Frequency")
        # ax2.plot(cnn_trough_idx*8, np.reciprocal(cnn_trough_period*8/sample_freq)*60, '+-', label="CNN Out Trough to Trough PFrequencyeriod")

        # ax2.plot(interp_breath_trough_period/sample_freq, label="Thermistor Trough to Trough Period")
        # ax2.plot(normalize(interp_ppg_peak_period), label="Smoothed period between peaks")
        plt.legend()
        plt.xlabel("Samples (at 200Hz)")
        plt.ylabel("RR in bpm")
        plt.title("Predicted vs Thermistor RR`")
        plt.show()


    """

    n = ppg_filtered.size//3
    breath_fft = np.fft.fft(breath_filtered[:n])
    ppg_peak_period_fft = np.fft.fft(interp_ppg_peak_period[:n])
    ppg_trough_period_fft = np.fft.fft((ppg_peak_envelope-ppg_trough_envelope)[:n])

    # max_bin = int(n/sample_freq/2*120)
    max_bin = int(200/60/sample_freq*n)

    ax2.plot(60*sample_freq/n*np.arange(breath_fft.size)[1:max_bin], normalize(np.abs(breath_fft)[1:max_bin]))
    # ax2.plot(60*sample_freq/n*np.arange(breath_fft.size)[1:max_bin], np.abs(ppg_peak_period_fft)[1:max_bin])
    # ax2.plot(60*sample_freq/n*np.arange(breath_fft.size)[1:max_bin], np.abs(ppg_trough_period_fft)[1:max_bin])
    ax2.plot(60*sample_freq/n*np.arange(breath_fft.size)[1:max_bin], normalize(np.abs(ppg_trough_period_fft)[1:max_bin]))
    """


    """
    inhale_period = np.empty_like(inhale_idx)
    inhale_period[1:] = np.diff(inhale_idx)
    inhale_period[0] = inhale_period[1]
    inhale_period = inhale_period/sample_freq
    ax2.plot(inhale_idx/sample_freq, np.reciprocal(inhale_period)*60.0, 'r.')
    """

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
