from features.feature import IdentityWindowFeature, TrainableFeature
from features.simple_filter import *
from features.peak_feature import *
from features.envelope import *
from features.breath_cnn_feature import *
from utils.model import build_model
from utils.thermistor import instant_bpm
import argparse
import json
import h5py
import os

feature_dict = {
        "id" : IdentityWindowFeature,
        "ea" : WindowEnvelopesAmplitude,
        "ptp" : WindowPeakTroughPeriods,
        "butter" : SimpleButterFilter,
        "spline" : SimpleSplineFilter,
        "local_norm" : SimpleLocalNorm,
        "breath_cnn" : BreathCNNFeature
        }

squared_error =  lambda x,y: np.mean((x-y)**2)
metric_dict = {
        "squared_error" : squared_error
        }

visualization_dict = {
    "plot_resp_rate_vs_target" : None
    }

def train_model(model, sampling_rate, training_path, validation_path):
    train_x,train_y = load_dataset(training_path,sampling_rate)
    val_x,val_y = load_dataset(validation_path,sampling_rate)

    for feature in model.get_list():
        if isinstance(feature,TrainableFeature):
            feature.train(train_x,train_y,val_x,val_y)

"""
def test_model(test_json,model,sampling_rate)
    test_x,test_y = load_dataset(test_path,sampling_rate)

    test_json_data = json.load(test_json)
    outputs = [trained_model.calc_feature(x) for x in test_x]:w
    metrics = test_json_data["metrics"]
    visualizations  = test_json_data["visualizations"]

    for metric in metric:
        loss_func = metric_dict[metric]

        losses = []
        for i in range(len(test_x))
            losses.append(
            y = instant_bpm(sampling_rate,test_y)
            x = model.calc_feature(test_x):
            losses.append(loss_func(x,y)

        mean = np.mean(losses)
        std_dev = np.std(losses)
        print(metric,"\tMean:",mean,"\tStandard Deviation",std_dev)

    for vis,i in visualizations:
        visualization = visualizations_dict[vis]:
        visualization(model,test_x[i],test_y[i]]
"""

def load_dataset(path,sampling_rate):
    files = os.listdir(path)
    inputs = []
    targets = []
    for f in files:
        data = h5py.File(os.path.join(path, f), 'r')['data']

        input_signal = data['signal']
        target_signal = data['target']
        sampling_rate_signal = data.attrs['sampling_rate']
        assert sampling_rate == sampling_rate_signal
        inputs.append(input_signal)
        targets.append(target_signal)
    return inputs,targets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sampling_rate', type=int, help='the sampling rate of the data')
    parser.add_argument('model_json_path',  help='the path to the json description of the model')
    #parser.add_argument('test_json_path',  help='path to test description json')
    parser.add_argument('training_path',  help='path to the training data directory')
    parser.add_argument('validation_path',  help='path to the validation data directory')
    #parser.add_argument('test_path',  help='path to the test data directory')
    return parser.parse_args()

def main():
    args = parse_args()
    sampling_rate = args.sampling_rate
    model_json_path = args.model_json_path
    training_path = args.training_path
    validation_path = args.validation_path

    model = build_model(sampling_rate,model_json_path)
    train_model(model, sampling_rate, training_path, validation_path)

if __name__ == "__main__":
    main()

