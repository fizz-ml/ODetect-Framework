from features.feature import *
from features.simple_filter import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch as t
from torch import FloatTensor as FT
from torch.autograd import Variable as V
import os

class BreathCNNFeature(WindowFeature,TrainableFeature):
    def __init__(self,sampling_rate,parameter_dict):
        super(BreathCNNFeature,self).__init__(sampling_rate,parameter_dict)
        self._cuda = t.cuda.is_available()
        self._cnn = BreathCNN(
                len(self._in_features),
                self._channel_count,
                self._feild,
                self._dilation,
                self._layers,
                )
        if self._cuda:
            self._cnn.cuda()

    def calc_feature(self, window):
        inputs = self._process_inputs(window)
        output = self._cnn(inputs)[0,0,:].cpu().data.numpy()
        return np.resample(output,len(window))
    
    def _process_input(self,x_window):
        feature_data = []
        ds_window = x_window[::self._down_sample]
        for feature in self._in_features:
            feature_data.append(feature.calc_feature(ds_window))

        inputs  = V(FT(np.expand_dims(np.stack(feature_data),0)))
        if self._cuda:
            inputs = inputs.cuda()
        return inputs

    def _process_label(self, y_window):
        ds_window = y_window[::self._down_sample]
        data_n = ((ds_window-np.mean(ds_window))/np.std(ds_window))
        spline_filtered_data = SimpleSplineFilter(avg_win=int(20/200*self._sampling_rate),ds=int(20/200*self._sampling_rate),s=int(45/200*self._sampling_rate)).calc_feature(data_n)
        thermistor = SimpleLocalNorm(self._sampling_rate,{"local_window_lenght":40}).calc_feature(spline_filtered_data)
        label = V(FT(np.expand_dims(np.expand_dims((thermistor,0),0))))
        if self._cuda:
            label = label.cuda()
        return label

    
    def train(self,train_x,train_y,val_x,val_y):
        t.manual_seed(7)
        np.random.seed(7)

        data_x_train_list = [self._process_input(x) for x in train_x]
        data_y_train_list = [self._process_label(y) for y in train_y]

        def closure():
            opt.zero_grad()
            sq_err = 0
            length = 0
            l_p = 0
            for i in range(len(data_y_train_list)):
                    x = data_x_train_list[i]
                    y = data_y_train_list[i]
                    out = model(x)
                    mean = out[:,0,:]
                    log_std = out[:,1,:]
                    l_p += -t.sum(-(y-mean)**2/(2*t.exp(log_std*2))-1*log_std)
                    length += y.size()[2]
            loss = l_p/length
            loss.backward()
            return sq_err

        #TODO: implement early stopping
        for i in self._train_time:
            opt.step(closure)
            print(i)

        torch.save(self._cnn.state_dict,self._param_path)

    def set_params(self, params):
        super(BreathCNNFeature,self).set_params(params)
        if os.path.exists(self._param_path):
            self._cnn.load_state_dict(torch.load(self._param_path))

    def get_params(self):
        super(BreathCNNFeature,self).get_params()
        if not os.path.exists(self._param_path):
            torch.save(self._cnn.state_dict,self._param_path)

    def get_param_template(self):
        param_template = {
            "channel_count" : (int,"the number of channels in the hidden layers of the cnn"),
            "dilation" : (int,"the dilation of each factor in the cnn"),
            "feild" : (int,"the receptive feild of each layer in a cnn"),
            "layers" : (int,"the number of layers in the cnn"),
            "in_features" : (list,"list of feature objects that will go into the cnn"),
            "down_sample" : (int,"factor by which to down_sample the input signal"),
            "train_time" : (int,"how many iterations to train the cnn"),
            "learning_rate" : (int,"learning rate for the cnn"),
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
            k: receptive feild reccomend 11
            dil: dilations of the convolutions recommend 10
        """
        super(BreathCNN,self).__init__()

        assert dil % 2 == 0
        assert k % 2 == 1

        pad = (k-1)*dil//2
        self._layers = nn.ModuleList()
        self._layers.append(nn.Conv1d(6,n*2,dil-1,padding=dil//2-1))
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
    features = [IdentityWindowFeature(200,{})]
    param_template = {
        "channel_count" : 10,
        "dilation" : 10,
        "feild" : 15,
        "layers" : 4,
        "in_features" : features,
        "down_sample" : 12,
        "train_time" : 500,
        "learning_rate" : 0.002,
        "param_path" : "bla.t7"
    }
    BreathCNNFeature(200,param_template)
    train_data = [np.random.randn(100000)]*10
    test_data = [np.random.randn(100000)]*10
    BreathCNNFeature(200,param_template).train(train_data,test_data,None,None)


if __name__ == "__main__":
    main_test()
