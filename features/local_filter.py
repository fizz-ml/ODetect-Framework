from features.feature import WindowFeature
from features.simple_filter import SimpleButterFilter
from scipy import signal
import numpy as np


class LocalLowPassFilter(WindowFeature):
    '''
    Local low-pass filter that goes over signal in window_size windows
    
    '''
    def __init__(self, sample_freq, cutoff_multiple, order=5):
        self._sample_freq = sample_freq
        self._cutoff_multiple = cutoff_multiple
        self._order = order
    return    

    def calc_feature(self, x, window_size):
	n = x.shape[0]
	y=[]
	rang = n//window_size
	for i in range(rang):        
	    local_signal =  x[i*window_size:min((i+1)*window_size, n)]        
	    f, t, Zxx = signal.stft(local_signal, fs=self._sample_freq, nperseg=window_size/4, noverlap=window_size/5, boundary=None)
	    f_mean_coefficients = np.mean(np.abs(Zxx[2:]),axis=1)
	    f_max = f[np.argmax(f_mean_coefficients) + 2]
	    filt = SimpleButterFilter(self._sample_freq, 1/60, f_max*self._cutoff_multiple, order=self._order)
	    y.append(filt.calc_feature(local_signal))
	    
    return np.concatenate(y)
