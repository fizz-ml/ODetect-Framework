def plot_fourier_spectrum(x, sample_freq, window_size):
    '''
    Plots fourier spectrium of signal x, averaged over 
    '''
    
    f, t, Zxx = signal.stft(x, fs=sample_freq, nperseg=window_size/4, noverlap=window_size/5, boundary=None)
    f_mean_coefficients = np.mean(np.abs(Zxx[2:]),axis=1)
    return plt.plot(f[2:100]*60, f_mean_coefficients[2:100])
                
