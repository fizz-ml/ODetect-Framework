from features.feature import WindowFeature
from scipy.signal import butter, lfilter, zpk2sos, sosfilt, resample
from scipy import signal
import numpy as np
from scipy import interpolate

class SimpleSplineFilter(WindowFeature):
    """ Passes local average over the window then fits regularized cubic splines to it and outputs interpolated values. """
    def calc_feature(self, window):
        window = self._in_features[0].calc_feature(window)
        win_size = int(np.floor(float(self._local_window_length*self._sampling_rate)))
        w=np.ones(win_size,'d')
        window = np.convolve(w/w.sum(),window,mode='same')

        tck = interpolate.splrep(np.arange(window.size)[::self._ds], window[::self._ds], s=self._s)
        return interpolate.splev(np.arange(window.size), tck, der=0)

    def get_param_template(self):
        param_template = {
            "local_window_length": (float, "Window to use for applying local averaging before spline fitting in seconds."), # Default was 10/200
            "ds": (int, "Level of downsampling before fitting the spline. (Mostly to limit computation cost)"), # Try 10 or higher
            "s": (float, "Regularization parameter for smoothing the spline interpolation.") # Default was 1.0 depends on signal (upwards of 60.0 for breath)
            }
        return param_template

class SimpleButterFilter(WindowFeature):
    """ Runs a butterworth bandpass filter over the window. Expects fs, lowcut and highcut in Hz. """

    def calc_feature(self, window):
        window = self._in_features[0].calc_feature(window)
        return self._butter_bandpass_filter(window, self._low_cut, self._high_cut, self._sampling_rate, self._order)

    def _butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        z,p,k = butter(order, [low, high], btype='bandpass', output='zpk')
        sos = zpk2sos(z, p, k)
        return sos

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        sos = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos,data)
        return y

    def get_param_template(self):
        param_template = {
            "low_cut": (float, "The lower limit of the frequency band in Hz."), # Default was 3/60
            "high_cut": (float, "The upper limit of the frequency band in Hz."), # Default was 90/60
            "order": (int, "The order of the filter.") # Default was 3
            }
        return param_template

class SimpleLocalNorm(WindowFeature):
    """ Locally normalizes the input signal by removing mean and standard deviation computed locally around the current value weighted by a Hann window. """
    def calc_feature(self, window):
        window = self._in_features[0].calc_feature(window)
        win_size = int(np.floor(float(self._local_window_length*self._sampling_rate)))

        win = signal.hann(win_size)
        #TODO replace these with reflected padding for conv to get rid of edge effects
        sig_mean = signal.convolve(window, win, mode='same') / sum(win)
        shift_sig = window - sig_mean

        abs_sig = np.abs(shift_sig)
        win = signal.hann(win_size)
        sig_std = signal.convolve(abs_sig, win, mode='same') / sum(win)

        norm_sig = shift_sig/sig_std
        return norm_sig

    def get_param_template(self):
        param_template = {
            "local_window_length": (float, "The size of the Hanning window to compute local statistics with in seconds.") # Default was 2000/200
            }
        return param_template

def normalize(ppg_signal):
    ppg_signal = (ppg_signal-np.mean(ppg_signal, axis=0))/np.std(ppg_signal)
    return ppg_signal
