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

def visualize_models(input_path, model_jsons):
    # Parse the input data
    data = h5py.File(input_path, 'r')['data']

    input_signal = data['signal']
    target_signal = data['target']
    sampling_rate = data.attrs['sampling_rate']
    ts = np.linspace(0, input_signal.size/sampling_rate, input_signal.size)

    # Reconstruct the models
    models = [build_model(sampling_rate, x) for x in model_jsons]

    models_out = []
    for model in models:
        models_out.append(model(input_signal))

    # Visualize raw outputs
    fig, ax = plt.subplots(1,1)
    target_signal = normalize(target_signal)
    filtered_target = SimpleSplineFilter(sampling_rate, [], {'local_window_length':60/200,'ds':20,'s':45}).calc_feature(target_signal)
    filtered_target = SimpleLocalNorm(sampling_rate, [], {"local_window_length":40}).calc_feature(filtered_target)
    ax.set_title("Model Output")
    for i, model in enumerate(models):
        ax.plot(ts, models_out[i], label=model.name)
    ax.plot(ts, filtered_target, label="Filtered Thermistor")
    ax.set_xlabel("Time in seconds")
    plt.legend()
    plot_max()

    fig, ax = plt.subplots(1,1)
    bpm = thermistor.instant_bpm(target_signal, sampling_rate, interpolate=False)
    ax.plot(bpm[0], bpm[1], '+-', linewidth=0.5, label="Thermistor RR")

    for i, model in enumerate(models):
        predicted = thermistor.instant_bpm(models_out[i], sampling_rate, interpolate=False)
        ax.plot(predicted[0], predicted[1], '+-', linewidth=0.5, label=model.name)

    plt.legend()
    plt.xlabel("Time in s")
    plt.ylabel("RR in bpm")
    plt.title("Predicted vs Thermistor RR")
    plot_max()


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_path', type=str, help='Path to h5 file to be visualized.')
    parser.add_argument('--models', nargs='+', help='List of model json paths', required=True)
    # parser.add_argument('thing', type=int, help='Name of the dataset under raw containing the data folder of h5 files to be processed.')
    args = parser.parse_args()

    model_jsons = args.models
    input_path = args.data_path

    visualize_models(input_path, model_jsons)
