import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

def filter_signal(signal,sampling_rate,high_freq,low_freq):
    n  = len(signal)
    hf_index = int(np.ceil(n*high_freq/sampling_rate/2))
    lf_index = int(np.floor(n*low_freq/sampling_rate/2))
    signal_fft = fft(signal)
    filter_fft = signal_fft
    filter_fft[hf_index:] = 0
    filter_fft[:lf_index] = 0
    filtered_signal = ifft(filter_fft)
    return filtered_signal 

def main():
    data = np.loadtxt("../data/exp_009.csv", delimiter =",")[:,1]
    data_n = (data-np.mean(data))/np.std(data)*3
    data_n = -1*filter_signal(data_n,200,3,0.05)
    plt.plot(data_n)
    data = np.loadtxt("../data/exp_009.csv", delimiter =",")[:,3]
    data_n = (data-np.mean(data))/np.std(data)

    fs = filter_signal(data_n,200,1,0.1)
    a = fs[:-2]
    b = fs[1:-1]
    c = fs[2:]
    peaks = np.array(np.where(np.logical_and(a < b, b > c) != 0))+1
    troughs = np.array(np.where(np.logical_and(a > b, b < c) != 0))+1
    peak_amplitudes = fs[peaks]
    troughs_amplitudes = fs[troughs]
    plt.plot(fs)
    plt.scatter(peaks,peak_amplitudes)
    plt.scatter(troughs,troughs_amplitudes)
    plt.show()

if __name__ == "__main__":
    main()

