from peak_detector import filter_signal
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

def get_x():
    data = np.loadtxt("../data/exp_009.csv", delimiter =",")[:,1]
    data_n = (data-np.mean(data))/np.std(data)
    fs = filter_signal(data_n,200,7,0.1)
    real_fs = np.real(fs)
    return np.expand_dims(np.expand_dims(real_fs,0),0)

def get_y():
    data = np.loadtxt("../data/exp_009.csv", delimiter =",")[:,3]
    data_n = (data-np.mean(data))/np.std(data)
    fs = filter_signal(data_n,200,1,0.5)
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
        self.layers.append(nn.Conv1d(1,n,9,padding=4))
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
    model = BreathCNN().cuda()
    opt = optim.Adam(model.parameters(),lr = 0.0004)

    data_x_train = V(FT(get_x()[:,:,:100000:8])).cuda()
    data_y_train = V(FT(get_y()[:,:,:100000:8])).cuda()
    data_x_test = V(FT(get_x()[:,:,100001:-1:8])).cuda()
    data_y_test = V(FT(get_y()[:,:,100001:-1:8])).cuda()
    def closure():
        opt.zero_grad()
        sq_err = t.mean((data_y_train - model(data_x_train))**2)
        sq_err.backward()
        sq_err_test = t.mean((data_y_test - model(data_x_test))**2)
        print(sq_err.cpu().data.numpy(),"\t",sq_err_test.cpu().data.numpy())
        return sq_err


    for i in range(500):
        opt.step(closure)

    plt.plot(model(data_x_test).cpu().data.numpy()[0,0,:])
    plt.plot(data_y_test.cpu().data.numpy()[0,0,:])
    plt.show()
        
if __name__ == "__main__":
    main()


        
    
