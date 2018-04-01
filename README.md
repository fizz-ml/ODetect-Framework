# ODetect-Framework

Data Collection

We include a simple script to extract simultaneous max sensor and thermistor data.


The run-time behaviour is specified in the config file as follows:


algorithms: specify a list of algorithms from the algorithms/ folder to use
params: for each algorithm, add a corresponding list of parameteres for that algorithm (parameter requirments for each algorithm are specified in the file for each algorithm)
run-behavious: 'evaluate' or 'real-time'


Example config file:

'''
algorithm: ['stft']
params: [[10, 5]]
run-behaviour: 'evaluate'


'''


