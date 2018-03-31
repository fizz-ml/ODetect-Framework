from scipy import signal
import numpy as np

def plot_fourier_spectrum(x, sample_freq, window_size):
    '''
    Plots Fourier Spectruum of signal, averaged over windows of window_size
    '''
    f, t, Zxx = signal.stft(x, fs=sample_freq, nperseg=window_size, noverlap=window_size*0.9, boundary=None)
    f_mean_coefficients = np.mean(np.abs(Zxx[2:]),axis=1)
    plt.plot(f[2:100]*60, f_mean_coefficients[2:100])
    plt.title('averaged fourier coefficients over ' + str(window_size/sample_freq) + ' second windows')
    plt.xlabel('frequency(bpm)')
    plt.ylabel('fourier coefficients')
    return           
