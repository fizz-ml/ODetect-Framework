import numpy as np
from features.simple_filter import SimpleLocalNorm, SimpleSplineFilter, normalize
from features.peak_feature import OldPeakTroughPoints

def peak_trough_points(breath_signal, sample_freq=200):
    breath_signal = normalize(breath_signal)

    breath_filtered = SimpleSplineFilter(sample_freq, [], {'local_window_length': 60/200, 'ds': 20, 's': 45.0}).calc_feature(breath_signal)
    breath_filtered = SimpleLocalNorm(sample_freq, [], {'local_window_length':10000}).calc_feature(breath_filtered)
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

def instant_breathing_period(breath_signal, sample_freq=200):
    ((breath_peak_idx, breath_peak_val, breath_peak_period),(breath_trough_idx, breath_trough_val, breath_trough_period)) = peak_trough_points(breath_signal, sample_freq)
    idxs = np.concatenate((breath_peak_idx, breath_trough_idx))
    periods = np.concatenate((breath_peak_period, breath_trough_period))

    # Need to sort them in time
    sort_idxs = np.argsort(idxs)
    idxs = idxs[sort_idxs]
    periods = periods[sort_idxs]

    print(periods)
    return np.interp(np.arange(breath_signal.size), idxs, periods)

def instant_bpm(breath_signal, sample_freq=200):
    return np.reciprocal(instant_breathing_period(breath_signal, sample_freq)/sample_freq)*60

