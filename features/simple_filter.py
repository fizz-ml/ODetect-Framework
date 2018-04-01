from features.feature import WindowFeature
from scipy.signal import butter, lfilter, zpk2sos, sosfilt, resample
from scipy import signal
import numpy as np
from scipy import interpolate

class SimpleSplineFilter(WindowFeature):
    def __init__(self, avg_win=10, ds=10, s=1.0):
        self._avg_win=avg_win
        self._ds=ds
        self._s=s

    def calc_feature(self, window):
        w=np.ones(self._avg_win,'d')
        breath_filtered1 = np.convolve(w/w.sum(),window,mode='same')

        tck = interpolate.splrep(np.arange(breath_filtered1.size)[::self._ds], breath_filtered1[::self._ds], s=self._s)
        breath_filtered = interpolate.splev(np.arange(breath_filtered1.size), tck, der=0)

        return breath_filtered

class SimpleButterFilter(WindowFeature):
    def __init__(self, fs, lowcut, highcut, order=2):
        """
        Expects fs, lowcut and highcut in Hz
        """
        self._fs = fs
        self._lowcut = lowcut
        self._highcut = highcut
        self._order = order

    def calc_feature(self, window):
        return self._butter_bandpass_filter(window, self._lowcut, self._highcut, self._fs, order=self._order)

    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        z,p,k = butter(order, [low, high], btype='bandpass', output='zpk')
        sos = zpk2sos(z, p, k)
        return sos

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos,data)
        return y

class SimpleLocalNorm(WindowFeature):
    """ Locally normalizes the input signal by removing mean and standard deviation computed locally around the current value weighted by a Hann window. """
    def calc_feature(self, window):
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
