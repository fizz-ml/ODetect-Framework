from features.feature import WindowFeature
from scipy.signal import butter, lfilter
from scipy import signal
import numpy as np
from scipy import interpolate

class SimpleSplineFilter(WindowFeature):
    def __init__(self):
        pass

    def calc_feature(self, window):
        w=np.ones(20,'d')
        breath_filtered1 = np.convolve(w/w.sum(),window,mode='same')

        tck = interpolate.splrep(np.arange(breath_filtered1.size)[::40], breath_filtered1[::40], s=25.0)
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
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

#TODO: Saturday I will call this something else thats more professional or someshit like that
def stupid_local_norm(sig, win_size=2000):
    win = signal.hann(win_size)
    sig_mean = signal.convolve(sig, win, mode='same') / sum(win)
    shift_sig = sig - sig_mean

    abs_sig = np.abs(shift_sig)
    win = signal.hann(win_size)
    sig_std = signal.convolve(abs_sig, win, mode='same') / sum(win)

    norm_sig = shift_sig/sig_std
    return norm_sig

