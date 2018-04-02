from features.feature import WindowFeature
from features.simple_filter import SimpleButterFilter
from scipy import signal
import numpy as np


class LocalLowPassFilter(WindowFeature):
    '''
    Low-pass filter that filters signal based on localized window frequencies
    
    '''
    def __init__(self, sample_freq, cutoff_multiple, order=5):
        """Intializes the filter
        
        Keyword arguments:
        sample_freq -- sample frequency (float)
        cutoff_multiple -- multiple of the strongest frequency to filter above (must be greater than 1.0)
        order -- order of butter filter (default 5)

        """
        self._sample_freq = sample_freq
        self._cutoff_multiple = cutoff_multiple
        self._order = order
    return    

    def calc_feature(self, x, window_size):
	"""Calculates feature on given signal

        Keyword arguments:
        x -- signal (1-d numpy array)
        window_size -- sample length over which each filter is applied (int, must be less than length of x)
        noverlap -- overlap between consecutive filter windows (int, must be less than window_size)
        """
        
        n = x.shape[0]
	y=[]
	rang = n//window_size
	for i in range(rang+1):        
	    local_signal =  x[i*window_size:min((i+1)*window_size, n)]
	    if (i == rang):
	    	window_size = min(n,(i+1)*window_size) - i*window_size - 1
                noverlap = int(noverlap/2) 
	    f, t, Zxx = signal.stft(local_signal, fs=self._sample_freq, nperseg=window_size, noverlap=noverlap, boundary=None)
	    f_mean_coefficients = np.mean(np.abs(Zxx[2:]),axis=1)
	    f_max = f[np.argmax(f_mean_coefficients) + 2]
	    filt = SimpleButterFilter(self._sample_freq, 1/60, f_max*self._cutoff_multiple, order=self._order)
	    y.append(filt.calc_feature(local_signal))
	    
    return np.concatenate(y)
