import algorithms.algorithm as algorithm

import utils.buffer

import scipy.fftpack as fft
import numpy as np

class SFTModel(algorithm.Algorithm):
    def __init__(self, params):
        super().__init__(params)
        self._window_size = int(self._window_length * self._fs)
        self._buffer = utils.buffer.Buffer(self._window_size)

    def update(self, x):
        """Updates the model with a new data point"""
        self._update_buffer(x)

    def evaluate(self):
        """Return the next predicted point"""
        b = self._get_buffer()
        max_bin = np.ceil(self._max_freq / self._fs * self._window_size).astype(np.int32)
        min_bin = np.floor(self._min_freq / self._fs * self._window_size).astype(np.int32)
        fft_bin = np.argmax(np.abs(fft.fft(b))[min_bin:max_bin]) + min_bin
        y = fft_bin * self._window_length # self._fs / self.window_size
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
            "fs": (float, "Input signal sampling rate in Hz."),
            "window_length": (float, "Length of buffer in seconds."),
            "max_freq": (float, "The maximum expected RR in Hz."),
            "min_freq": (float, "The minimum expected RR in Hz.")
            }
        return param_template

