import algorithms.algorithm as algorithm

import utils.buffer

import scipy.fftpack as fft
import numpy as np

SAMPLING_RATE = 300.0
class SFTModel(algorithm.Algorithm):
    def __init__(self, params):
        super().__init__(params)
        self._buffer = utils.buffer.Buffer(self._window_length)
        self._max_freq = 0.6
        self._min_freq = 0.05

    def __call__(self,x):
        """Return the next predicted point"""
        self._update_buffer(x)
        b = self._get_buffer()
        max_bin = np.ceil(self._max_freq / SAMPLING_RATE * self.window_size).astype(np.int32)
        min_bin = np.floor(self._min_freq / SAMPLING_RATE * self.window_size).astype(np.int32)
        fft_bin = np.argmax(np.abs(fft.fft(b))[min_bin:max_bin]) + min_bin
        y = fft_bin * SAMPLING_RATE / self.window_size
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
            "window_length": (float, "Length of buffer in seconds."),
            "max_freq": (float, "The maximum expected RR in Hz."),
            "min_freq": (float, "The minimum expected RR in Hz.")
            }
        return param_template

