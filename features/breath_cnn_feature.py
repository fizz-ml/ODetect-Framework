from features.feature import *
from features.simple_filter import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch as t
import torch.optim as optim
from torch import FloatTensor as FT
from torch.autograd import Variable as V
from scipy.signal import resample
import os
import copy

class BreathCNNFeature(WindowFeature,TrainableFeature):
    def __init__(self,sampling_rate,in_features,parameter_dict):
        self._cuda = t.cuda.is_available()
        super(BreathCNNFeature,self).__init__(sampling_rate,in_features,parameter_dict)
        if self._cuda:
            self._cnn.cuda()

    def calc_feature(self, window):
        inputs = self._process_input(window)
        output = self._cnn(inputs)[0,0,:].cpu().data.numpy()
        return resample(output,len(window))

    def _process_input(self,x_window):
        feature_data = []
        for feature in self._in_features:
            feature_data.append(feature.calc_feature(x_window)[::self._down_sample])

        inputs  = V(FT(np.expand_dims(np.stack(feature_data),0)))
        if self._cuda:
            inputs = inputs.cuda()
        return inputs

    def _process_label(self, y_window):
        y_window = normalize(y_window)
        spline_filtered_data = SimpleSplineFilter(self._sampling_rate, [], {'local_window_length':20/200,'ds':20,'s':45}).calc_feature(y_window)
        thermistor = SimpleLocalNorm(self._sampling_rate, [], {"local_window_length":40}).calc_feature(spline_filtered_data)
        thermistor = thermistor[::self._down_sample]
        import matplotlib.pyplot as plt
        plt.plot(thermistor)
        plt.plot(spline_filtered_data)
        plt.plot(thermistor)
        plt.show()
        label = V(FT(np.expand_dims(np.expand_dims(thermistor,0),0)))
        if self._cuda:
            label = label.cuda()
        return label


    def train(self,train_x,train_y,val_x,val_y):
        t.manual_seed(7)
        np.random.seed(7)
        opt = optim.Adam(self._cnn.parameters(),lr=self._learning_rate)

        data_x_train_list = [self._process_input(x) for x in train_x]
        data_y_train_list = [self._process_label(y) for y in train_y]

        data_x_val_list = [self._process_input(x) for x in val_x]
        data_y_val_list = [self._process_label(y) for y in val_y]

        def closure():
            opt.zero_grad()
            sq_err = 0
            length = 0
            l_p = 0
            for i in range(len(data_y_train_list)):
                x = data_x_train_list[i]
                y = data_y_train_list[i]
                out = self._cnn(x)
                mean = out[:,0,:]
                log_std = out[:,1,:]
                l_p += -t.sum(-(y-mean)**2/(2*t.exp(log_std*2))-1*log_std)
                length += y.size()[2]
            loss = l_p/length

            sq_err_val = 0
            length_val = 0
            for i in range(len(data_y_val_list)):
                x = data_x_val_list[i]
                y = data_y_val_list[i]
                out = self._cnn(x)
                mean = out[:,0,:]
                log_std = out[:,1,:]
                sq_err_val += -t.sum(-(y-mean)**2/(2*t.exp(log_std*2))-1*log_std)
                length_val += y.size()[2]
            sq_err_val = sq_err_val/length_val
            # sq_err_test = -t.sum(-(y-mean)**2/(2*t.exp(log_std*2))-1*log_std)

            print("Train: ", loss.cpu().data.numpy(),"\t","Test: ", sq_err_val.cpu().data.numpy())

            loss.backward()


            l = [x.grad.cpu() for x in self._cnn.parameters()]
            return loss, sq_err_val

        # Keep track lowest validation error and associated state
        min_val_err = 99999999999 #TODO: This needs to be infinity, but giving casting error
        min_loss_state = None
        for i in range(self._train_time):
            print('Minvalerr', min_val_err)
            loss, val_err = closure()
            if val_err.cpu().data.numpy()[0] < min_val_err:
                min_val_err = val_err.cpu().data.numpy()[0]
                min_loss_state = copy.deepcopy(self._cnn.state_dict())
            opt.step()

        # Load the minimum validaiton error state back up
        self._cnn.load_state_dict(min_loss_state)
        t.save(self._cnn.state_dict(),self._param_path)

    def set_params(self, params):
        super(BreathCNNFeature,self).set_params(params)
        self._cnn = BreathCNN(
                len(self._in_features),
                self._channel_count,
                self._field,
                self._dilation,
                self._layers,
                )
        if os.path.exists(self._param_path):
            self._cnn.load_state_dict(t.load(self._param_path))

    def get_params(self):
        params = super(BreathCNNFeature,self).get_params()
        if not os.path.exists(self._param_path):
            t.save(self._cnn.state_dict(),self._param_path)
        return params

    def get_param_template(self):
        param_template = {
            "channel_count" : (int,"the number of channels in the hidden layers of the cnn"),
            "dilation" : (int,"the dilation of each factor in the cnn"),
            "field" : (int,"the receptive field of each layer in a cnn"),
            "layers" : (int,"the number of layers in the cnn"),
            "down_sample" : (int,"factor by which to down_sample the input signal"),
            "train_time" : (int,"how many iterations to train the cnn"),
            "learning_rate" : (float,"learning rate for the cnn"),
            "param_path" : (str, "path in which to save parameters")
        }
        return param_template


class BreathCNN(nn.Module):
    def __init__(self,i,n,k,dil,layers):
        """
        Initialize a breath cnn
        Args:
            i: number of inputs
            n: number of channels reccomend 15
            k: receptive field reccomend 11
            dil: dilations of the convolutions recommend 10
        """
        super(BreathCNN,self).__init__()

        assert dil % 2 == 0
        assert k % 2 == 1

        pad = (k-1)*dil//2
        self._layers = nn.ModuleList()
        self._layers.append(nn.Conv1d(i,n*2,dil-1,padding=dil//2-1))
        self._layers.append(nn.Conv1d(n*2,n,dil-1,padding=dil//2-1))
        for i in range(layers):
            self._layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
            self._layers.append(nn.Conv1d(n,n,1))
        self.out = nn.Conv1d(n,2,1)

    def __call__(self,x):
        h = x
        for layer in self._layers:
            h = f.relu(layer(h))
        out = self.out(h)
        return out

def main_test():
    np.random.seed(7)
    features = [IdentityWindowFeature(200,{})]
    param_template = {
        "channel_count" : 10,
        "dilation" : 10,
        "field" : 15,
        "layers" : 4,
        "in_features" : features,
        "down_sample" : 12,
        "train_time" : 5,
        "learning_rate" : 0.0002,
        "param_path" : "bla.t7"
    }
    BreathCNNFeature(200,param_template)
    train_data = [np.random.randn(100000)]*10
    test_data = [np.random.randn(100000)]*10
    BreathCNNFeature(200,param_template).train(train_data,test_data,None,None)
    BreathCNNFeature(200,param_template).get_params()
    assert len(BreathCNNFeature(200,param_template).calc_feature(np.random.randn(100000))) == 100000


if __name__ == "__main__":
    main_test()
