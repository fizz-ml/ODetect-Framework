import numpy as np
from features.simple_filter import SimpleLocalNorm, SimpleSplineFilter, normalize
from features.peak_feature import OldPeakTroughPoints
from scipy import signal

def peak_trough_points(breath_signal, sample_freq=200):
    breath_signal = normalize(breath_signal)

    breath_filtered = SimpleSplineFilter(sample_freq, [], {'local_window_length': 60/200, 'ds': 20, 's': 45.0}).calc_feature(breath_signal)
    breath_filtered = SimpleLocalNorm(sample_freq, [], {'local_window_length':10000/200}).calc_feature(breath_filtered)
    breath_filtered = normalize(breath_filtered)

    ((breath_peak_idx, breath_peak_val, breath_peak_period),(breath_trough_idx, breath_trough_val, breath_trough_period)) = OldPeakTroughPoints().calc_feature(breath_filtered, delta=1.0, lookahead=100)

    return ((breath_peak_idx, breath_peak_val, breath_peak_period),(breath_trough_idx, breath_trough_val, breath_trough_period))

def breathing_phase(breath_signal, sample_freq=200):
    ((breath_peak_idx, breath_peak_val, breath_peak_period),(breath_trough_idx, breath_trough_val, breath_trough_period)) = peak_trough_points(breath_signal, sample_freq)
    idxs = np.concatenate((breath_peak_idx, breath_trough_idx))
    vals = np.concatenate((np.ones(breath_peak_idx.size), -1 * np.ones(breath_trough_idx.size)))

    # Need to sort them in time
    sort_idxs = np.argsort(idxs)
    idxs = idxs[sort_idxs]
    vals = vals[sort_idxs]

    return np.interp(np.arange(breath_signal.size), idxs, vals)

def instant_breathing_period(breath_signal, sample_freq=200, interpolate=True):
    ((breath_peak_idx, breath_peak_val, breath_peak_period),(breath_trough_idx, breath_trough_val, breath_trough_period)) = peak_trough_points(breath_signal, sample_freq)
    idxs = np.concatenate((breath_peak_idx, breath_trough_idx))
    periods = np.concatenate((breath_peak_period, breath_trough_period))

    # Need to sort them in time
    sort_idxs = np.argsort(idxs)
    idxs = idxs[sort_idxs]
    periods = periods[sort_idxs]

    if interpolate:
        return np.interp(np.arange(breath_signal.size), idxs, periods/sample_freq)
    else:
        return (idxs/sample_freq, periods/sample_freq)

def instant_bpm(breath_signal, sampling_rate=200, interpolate=True):
    if interpolate:
        return np.reciprocal(instant_breathing_period(breath_signal, sampling_rate))*60
    else:
        (idxs, periods) = instant_breathing_period(breath_signal, sampling_rate, interpolate=False)
        return (idxs, np.reciprocal(periods)*60)

def stft_centroid_bpm(input_signal, sampling_rate=200):
    max_freq = 30/60
    downsample = 2

    bf,bt, bZxx = signal.stft(input_signal[::downsample], fs=sampling_rate/downsample, nperseg=8000//downsample, noverlap=8000//downsample-10, boundary=None)
    max_bin = np.searchsorted(bf, max_freq)
    min_bin = 2
    bcentroid = centroid(np.log(1+np.abs(bZxx))[min_bin:max_bin], bf[min_bin:max_bin], axis = 0, p=3)
    bcentroid = np.interp(np.arange(input_signal.size)/200, bt, bcentroid)
    return bcentroid*60

def centroid(Z, f, axis=0, p=1):
    Z = np.power(Z, p)
    a = f
    if(axis==0):
        centroid = np.dot(np.transpose(Z),a)/np.sum(Z, axis=axis)
    elif(axis==1):
        centroid = np.dot(Z,a)/np.sum(Z,axis=axis)
    return centroid
