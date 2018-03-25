from models.sfft import SFFTModel
import argparse
from glob import glob
import numpy as np
import os
import evaluation.loss_functions as loss_functions
import h5py
import matplotlib.pyplot as plt

def evaluate_model(model, loss_func, data):
    """ Evaluates the given model with the given loss function on the given sample. """
    # Max length #TODO: just for testing
    max_length = 40000
    # Extract signals from the data
    ppg_signal = np.squeeze(data['signal']['pleth']['y'][:max_length])
    inhale_idx = np.squeeze(data['labels']['co2']['startinsp']['x'])
    sampling_rate = 300.0 #TODO: Actually extract this

    # Compute labels
    print(inhale_idx)
    rr_label = loss_functions.CO2_timestamptofreq(inhale_idx, sampling_rate)

    # Query model for predictions
    predictions = np.empty_like(ppg_signal)
    for i, x in enumerate(ppg_signal):
        predictions[i] = model(x)

    plt.plot(np.arange(predictions.shape[0]), predictions)
    plt.plot(inhale_idx, rr_label)
    plt.show()

    # Extract predictions at the peaks
    sampled_predictions = predictions[inhale_idx.astype(np.int32)]

    # Compare the loss
    return loss_func(rr_label, sampled_predictions)

def evaluate_dataset(model, loss_func, dataset_path):
    evals = []
    for sample_path in glob(os.path.join(dataset_path, '*.mat')):
        with h5py.File(sample_path, 'r') as f:
            loss = evaluate_model(model, loss_func, f)
            evals.append(loss)
            print(loss)
            print(sample_path)
    return sum(evals)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset under raw containing the data folder of h5 files to be processed.')
    args = parser.parse_args()

    dataset_name = args.dataset_name

    # Load some data
    input_path = os.path.join('data', dataset_name, 'raw')

    # Instantiate model
    hyperparams = 18000
    model = SFFTModel(hyperparams)

    # Evaluate
    print(evaluate_dataset(model, loss_functions.RMSE, input_path))

