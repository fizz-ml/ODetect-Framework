from utils.peak_detector import filter_signal
from features.envelope import *
from features.peak_feature import *
from features.simple_filter import *
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch import FloatTensor as FT
from torch.autograd import Variable as V
from scipy.signal import resample

t.manual_seed(7)
np.random.seed(7)

cpu_only = False

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
    fs6 = stupid_local_norm(data_n)
    #plt.plot(fs2[100001:-1:8])
    features = np.stack([fs3,fs1,fs4,fs5,fs6])
    return np.expand_dims(features,0)

def get_y(path,window):
    data = np.loadtxt(path, delimiter =",")[:,3]
    data_n = ((data-np.mean(data))/np.std(data))[window]
    fs0 = SimpleButterFilter(200,3/60,90/60,order=3).calc_feature(data_n)
    fs0 = SimpleSplineFilter(avg_win=20,ds=20,s=45).calc_feature(data_n)
    #fs1 = ((fs0-np.mean(fs0))/np.std(fs0))
    fs2 = stupid_local_norm(fs0,8000)
    real_fs = np.real(fs2)
    return np.expand_dims(np.expand_dims(fs2,0),0)

class BreathCNN(nn.Module):
    def __init__(self):
        super(BreathCNN,self).__init__()
        n = 15
        k = 11
        dil = 10
        pad = (k-1)*dil//2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(5,n*2,9,padding=4))
        self.layers.append(nn.Conv1d(n*2,n,9,padding=4))
        self.layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
        self.layers.append(nn.Conv1d(n,n,1))
        self.layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
        self.layers.append(nn.Conv1d(n,n,1))
        self.layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
        self.layers.append(nn.Conv1d(n,n,1))
        self.layers.append(nn.Conv1d(n,n,k,dilation=dil,padding=pad))
        self.layers.append(nn.Conv1d(n,n,1))
        self.out = nn.Conv1d(n,2,1)

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
    opt = optim.Adam(model.parameters(),lr = 0.0002)
    ds = 12

    paths = [
            "data/max/exp_007.csv",
            "data/max/exp_008.csv",
            "data/max/exp_009.csv",
            "data/max/exp_010.csv",
            "data/max/exp_011.csv",
            "data/max/exp_012.csv",
            "data/max/exp_013.csv",
            "data/max/exp_014.csv",
            "data/max/exp_015.csv",
            "data/max/exp_016.csv",
            "data/max/exp_017.csv",
            "data/max/exp_018.csv",
            "data/max/exp_019.csv",
            #"data/max/exp_020.csv",
            "data/max/exp_021.csv",
            "data/max/exp_022.csv",
            "data/max/exp_023.csv",
            "data/max/exp_024.csv",
            "data/max/exp_025.csv",
            #"data/max/exp_026.csv",
            "data/max/exp_027.csv",
            "data/max/exp_028.csv",
            "data/max/exp_029.csv",
            #"data/max/exp_030.csv",
            "data/max/exp_031.csv",
            ]
    data_x_train_list = []
    data_y_train_list = []
    for path in paths:
        data_x_train_list.append(V(FT(get_x(path,slice(0,-1))[:,:,::ds])))
        data_y_train_list.append(V(FT(get_y(path,slice(0,-1))[:,:,::ds])))
    data_x_test = V(FT(get_x(sys.argv[1],slice(0,-1))[:,:,::ds]))
    data_y_test = V(FT(get_y(sys.argv[1],slice(0,-1))[:,:,::ds]))


    l = data_y_test.size()[2]
    rsw = np.sin(np.linspace(0,420,l))/3
    crsw = V(FT(rsw))

    if not cpu_only:
        data_x_train_list = [x.cuda() for x in data_x_train_list]
        data_y_train_list = [y.cuda() for y in data_y_train_list]
        data_x_test = data_x_test.cuda()
        data_y_test = data_y_test.cuda()
        model = model.cuda()
        crsw = crsw.cuda()

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
        sq_err_test = t.mean((data_y_test - model(data_x_test)[0,0,:])**2)
        print(loss.cpu().data.numpy(),"\t",sq_err_test.cpu().data.numpy())
        return sq_err


    for i in range(500):
        opt.step(closure)
        print(i)

    mean = model(data_x_test).cpu().data.numpy()[0,0,:]
    log_std = model(data_x_test).cpu().data.numpy()[0,1,:]
    x1 = mean + np.exp(log_std)
    x2 = mean - np.exp(log_std)
    plt.plot(mean)
    plt.plot(log_std)
    plt.plot(log_std*0)
    plt.plot(data_y_test.cpu().data.numpy()[0,0,:])
    plt.show()


    x =  model(data_x_test).cpu().data.numpy()[0,0,:]
    print(len(x))
    print(len(resample(x,len(x)*ds)))
    return resample(x,len(x)*ds)

if __name__ == "__main__":
    main()




