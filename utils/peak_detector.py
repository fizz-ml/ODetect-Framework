import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

def filter_signal(signal,sampling_rate,high_freq,low_freq):
    n  = len(signal)
    hf_index = int(np.ceil(n*high_freq/sampling_rate))
    lf_index = int(np.floor(n*low_freq/sampling_rate))
    signal_fft = fft(signal)
    filter_fft = signal_fft
    filter_fft[hf_index:] = 0
    filter_fft[:lf_index] = 0
    filtered_signal = ifft(filter_fft)
    return filtered_signal 
def main():
    data = np.loadtxt("../data/exp_009.csv", delimiter =",")[:,3]
    fs = filter_signal(data,200,0.5,0.05)
    a = fs[:-2]
    b = fs[1:-1]
    c = fs[2:]
    peaks = np.array(np.where(np.logical_and(a < b, b > c) != 0))+1
    troughs = np.array(np.where(np.logical_and(a > b, b < c) != 0))+1
    peak_amplitudes = fs[peaks]
    troughs_amplitudes = fs[troughs]
    plt.plot(fs)
    plt.plot(data-np.mean(data))
    plt.scatter(peaks,peak_amplitudes)
    plt.scatter(troughs,troughs_amplitudes)
    plt.show()

if __name__ == "__main__":
    main()

