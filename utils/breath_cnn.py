from utils.peak_detector import filter_signal
from features.envelope import *
from features.peak_feature import *
from features.simple_filter import *
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch import FloatTensor as FT
from torch.autograd import Variable as V

t.manual_seed(7)
np.random.seed(7)

cpu_only = True

def get_x(path,window):
    data = np.loadtxt(path, delimiter =",")[:,1]
    data_n = ((data-np.mean(data))/np.std(data))[window]
    fs0 = SimpleButterFilter(200,3/60,90/60,order=3).calc_feature(data_n)
    fs1 = stupid_local_norm(fs0)
    fs2 = WindowEnvelopesAmplitude().calc_feature(fs1)
    fs3 = (fs2 - np.mean(fs2))/np.std(fs2)
    peakp,troughp = WindowPeakTroughPeriods().calc_feature(fs1)
    fs4 = (peakp - np.mean(peakp))/np.std(peakp)
    fs5 = (troughp - np.mean(troughp))/np.std(troughp)
    #plt.plot(fs2[100001:-1:8])
    features = np.stack([fs3,fs1,fs4,fs5])
    return np.expand_dims(features,0)

def get_y(path,window):
    data = np.loadtxt(path, delimiter =",")[:,3]
    data_n = ((data-np.mean(data))/np.std(data))[window]
    fs = SimpleButterFilter(200,3/60,90/60,order=3).calc_feature(data_n)
    real_fs = np.real(fs)
    return np.expand_dims(np.expand_dims(real_fs,0),0)

class BreathCNN(nn.Module):
    def __init__(self):
        super(BreathCNN,self).__init__()
        n = 10
        k = 11
        dil = 10
        pad = (k-1)*dil//2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(4,n,9,padding=4))
        self.layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
        self.layers.append(nn.Conv1d(n,n,1))
        """
        self.layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
        self.layers.append(nn.Conv1d(n,n,1))
        self.layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
        self.layers.append(nn.Conv1d(n,n,1))
        """
        self.layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
        self.layers.append(nn.Conv1d(n,n,1))
        self.out = nn.Conv1d(n,1,1)

    def __call__(self,x):
        h = x
        for layer in self.layers:
            h = f.relu(layer(h))
        out = self.out(h)
        #print(x.size())
        #print(out.size())
        return out


def main():
    model = BreathCNN()
    opt = optim.Adam(model.parameters(),lr = 0.0004)

    data_x_train = V(FT(get_x("data/max/exp_011.csv",slice(0,-1))[:,:,::8]))
    data_y_train = V(FT(get_y("data/max/exp_011.csv",slice(0,-1))[:,:,::8]))
    data_x_test = V(FT(get_x("data/max/exp_009.csv",slice(0,-1))[:,:,::8]))
    data_y_test = V(FT(get_y("data/max/exp_009.csv",slice(0,-1))[:,:,::8]))


    l = data_y_test.size()[2]
    rsw = np.sin(np.linspace(0,420,l))/3
    crsw = V(FT(rsw))

    if not cpu_only:
        data_x_train = data_x_train.cuda()
        data_y_train = data_y_train.cuda()
        data_x_test = data_x_test.cuda()
        data_y_test = data_y_test.cuda()
        model = model.cuda()
        crsw = crsw.cuda()

    def closure():
        opt.zero_grad()
        sq_err = t.mean((data_y_train - model(data_x_train))**2)
        sq_err.backward()
        sq_err_test = t.mean((data_y_test - model(data_x_test))**2)
        print(sq_err.cpu().data.numpy(),"\t",sq_err_test.cpu().data.numpy())
        return sq_err


    for i in range(65):
        opt.step(closure)
        print(i)

    """
    plt.plot(model(data_x_test).cpu().data.numpy()[0,0,:])
    plt.plot(data_y_test.cpu().data.numpy()[0,0,:])
    plt.show()
    """

    return model(data_x_test).cpu().data.numpy()[0,0,:]

if __name__ == "__main__":
    main()




