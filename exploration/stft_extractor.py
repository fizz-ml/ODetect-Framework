from utils.model import build_model
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import signal
from features.simple_filter import SimpleSplineFilter, SimpleLocalNorm, normalize
from utils import thermistor

def plot_max():
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

def visualize_stft_extractor(input_path, model_json):
    # Parse the input data
    data = h5py.File(input_path, 'r')['data']

    input_signal = data['signal']
    target_signal = data['target']
    sampling_rate = data.attrs['sampling_rate']
    ts = np.linspace(0, input_signal.size/sampling_rate, input_signal.size)

    # Reconstruct the model
    model = build_model(sampling_rate, model_json)
    model_out = model(input_signal)

    filtered_target = SimpleSplineFilter(sampling_rate, [], {'local_window_length':60/200,'ds':20,'s':45}).calc_feature(target_signal)
    filtered_target = SimpleLocalNorm(sampling_rate, [], {"local_window_length":40}).calc_feature(filtered_target)

    # Visualize STFT
    max_freq = 30/60
    downsample = 2

    f,t, Zxx = signal.stft(model_out[::downsample], fs=sampling_rate/downsample, nperseg=8000//downsample, noverlap=8000//downsample-10, boundary=None)
    bf,bt, bZxx = signal.stft(filtered_target[::downsample], fs=sampling_rate/downsample, nperseg=8000//downsample, noverlap=8000//downsample-10, boundary=None)
    max_bin = np.searchsorted(f, max_freq)
    min_bin = 2

    fig, ax = plt.subplots(2,1)
    ax[0].pcolormesh(bt, bf[min_bin:max_bin]*60, np.log(1+np.abs(bZxx)[min_bin:max_bin]))
    ax[1].pcolormesh(t, f[min_bin:max_bin]*60, np.log(1+np.abs(Zxx)[min_bin:max_bin]))

    # Argmax in frequency
    bmaxbins = np.argmax(np.abs(bZxx)[min_bin:max_bin], axis = 0) + min_bin
    # Histogram correction
    # ax[0].plot(bt, (bf[bmaxbins]+bf[bmaxbins+1])/2*60, 'r-')
    maxbins = np.argmax(np.abs(Zxx)[min_bin:max_bin], axis = 0) + min_bin
    # ax[1].plot(t, (f[maxbins]+f[maxbins+1])/2*60, 'r-')

    ax[0].set_title("Thermistor STFT P=3 Centroid")
    ax[1].set_title("Prediction STFT P=3 Centroid")
    ax[0].set_ylabel("RR in bpm")
    ax[1].set_xlabel("Time in s")
    ax[1].set_ylabel("RR in bpm")

    def centroid(Z, f, axis=0, p=1):
        Z = np.power(Z, p)
        a = f
        if(axis==0):
            centroid = np.dot(np.transpose(Z),a)/np.sum(Z, axis=axis)
        elif(axis==1):
            centroid = np.dot(Z,a)/np.sum(Z,axis=axis)
        return centroid

    # Centroid in frequency
    bcentroid = centroid(np.log(1+np.abs(bZxx))[min_bin:max_bin], bf[min_bin:max_bin], axis = 0, p=3)
    ax[0].plot(t, bcentroid*60, 'r-')
    centroid = centroid(np.log(1+np.abs(Zxx))[min_bin:max_bin], f[min_bin:max_bin], axis = 0, p=3)
    ax[1].plot(t, centroid*60, 'r-')
    """
    """

    plot_max()

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_path', type=str, help='Path to h5 file to be visualized.')
    parser.add_argument('model_json', type=str, help='Path to json of model to be used as the feature.')
    # parser.add_argument('thing', type=int, help='Name of the dataset under raw containing the data folder of h5 files to be processed.')
    args = parser.parse_args()

    model_json = args.model_json
    input_path = args.data_path

    visualize_stft_extractor(input_path, model_json)
