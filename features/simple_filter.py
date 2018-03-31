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

def normalize(ppg_signal):
    ppg_signal = (ppg_signal-np.mean(ppg_signal, axis=0))/np.std(ppg_signal)
    return ppg_signal
