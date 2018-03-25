import algorithms.algorithm as algorithm

import utils.buffer

import scipy.fftpack as fft
import scipy.signal.cwt as cwt
import scipy.signal.ricker as ricker
import numpy as np

SAMPLING_RATE = 300.0

class WaveletModel(algorithm.Algorithm):
    def __init__(self, params):
        self.set_params(params)
        self._buffer = utils.buffer.Buffer(self._params["window_length"])
        self._func = ricker
        self._widths = np.arange(self._params["min_width"], self._params["max_width"])
        
    def __call__(self,x):
        """Return the next predicted point"""
        
        self._update_buffer(x)
        b = self._get_buffer()        
        #y = fft_bin * SAMPLING_RATE / self.window_size
        
        y = cwt(b, self._func, self._widths)
        
        return y

    def reset():
        """Reset the model"""
        self._buffer = np.zeros(window_size)

    def _update_buffer(self,x):
        self._buffer = np.roll(self._buffer,1)
        self._buffer[0] = x

    def _get_buffer(self):
        return self._buffer

    def get_param_template(self):
        param_template = {
            "window_length": (int, "Length of buffer in seconds."),
            "min_width": (int, "Minimum width for wavelet function."),
            "max_width": (int, "Maximum width for wavelet function"),
            }
        return param_template

