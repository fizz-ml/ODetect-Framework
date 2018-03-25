import matplotlib.pyplot as plt
#import pandas
import numpy as np
data = np.genfromtxt('t3.csv', delimiter=',', names=['time', 'signal'])
data['time'] = data['time'] - data['time'][0]
#print(data['time'][1000] - data['time'][999])
diff = data['time'][1:] - data['time'][:-1]
#diff = data['signal'][1:] - data['signal'][:-1]

print(diff.shape)

p = np.empty_like(diff)
running_time = 0
for i in range(diff.shape[0]):
    running_time +=diff[i]
    p[i] =  running_time/i


#plt.plot(np.arange(diff.shape[0]), p)
plt.plot(np.arange(diff.shape[0]), diff)
#plt.plot(data['time'], data['signal'])
plt.show()
