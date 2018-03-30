import argparse
from glob import glob
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal

"""
Visualizes the short time fourier transform of the signal.
Produces graphs for each signal in the specified dataset.
The graphs are composed of one plot of the ppg and co2 waveforms and one plot of the 2D STFT spectrogram.
Red points on the spectrogram correspond to the equivalent instantaneous frequency derived from the labels.
"""

def visualize_dataset(dataset_path):
    for sample_path in glob(os.path.join(dataset_path, '*.mat')):
        print(sample_path)
        with h5py.File(sample_path, 'r') as data:
            max_length_load = 140000
            max_length = 3144
            sample_freq = 15
            ppg_signal = np.squeeze(data['signal']['pleth']['y'][0:max_length_load])[0:max_length_load]
            ppg_signal = ppg_signal[::20]
            ppg_signal = ppg_signal[0:max_length]

            # ppg_signal = signal.decimate(ppg_signal, 20)
            inhale_idx = np.squeeze(data['labels']['co2']['startinsp']['x'])

            # Plot
            fig, ax2 = plt.subplots(1,1)

            # Frequency
            max_bin = 160
            f,t, Zxx = signal.stft(ppg_signal, fs=sample_freq, nperseg=900, noverlap=890)
            ax2.set_xlabel("Time in s")
            ax2.set_ylabel("RR in bpm")
            ax2.pcolormesh(t, f[2:max_bin]*60, np.sqrt(np.abs(Zxx)[2:max_bin]))#/f[2:30, np.newaxis])
            ax2.set_ylim(f[2]*60, f[max_bin]*60)
            ax2.set_xlim(0, max_length/sample_freq)

            """
            inhale_period = np.empty_like(inhale_idx)
            inhale_period[1:] = np.diff(inhale_idx)
            inhale_period[0] = inhale_period[1]
            inhale_period = inhale_period/sample_freq
            ax2.plot(inhale_idx/sample_freq, np.reciprocal(inhale_period)*60.0, 'r.')
            """

            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset under raw containing the data folder of h5 files to be processed.')
    args = parser.parse_args()

    dataset_name = args.dataset_name

    # Load some data
    input_path = os.path.join('data', dataset_name, 'raw')
    print(visualize_dataset(input_path))
