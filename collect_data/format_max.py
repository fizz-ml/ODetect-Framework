"""
This script was written to collect simultaneous data from the max and thermistor sensors.
It is to be used in combination with max_thermistor.ino which is an Arduino script.
The script makes use of the pyserial module.
"""

import argparse
import time
import h5py
import os
import glob
import numpy as np

def calc_breath_signal(raw_max_therm):
    # Averages the two nostril values
    return (raw_max_therm[:,3].flatten()+raw_max_therm[:,2].flatten())/2

def format_data_file(input_path, output_path, sampling_rate=200):
    """ Formats a collected raw max thermistor file to standard data format. """
    try:
        # Cut off first 10 lines as max can be initializing
        raw_max_therm = np.genfromtxt(input_path, delimiter=',')[10:]
    except:
        print("Reading of raw max file {} failed".format(input_path))
        raise

    breath_signal = calc_breath_signal(raw_max_therm)
    ppg_signal = raw_max_therm[:,1].flatten()

    # This timestamp can be unreliable due to buffering (We assume constant sampling rate)
    unix_timestamp = raw_max_therm[:,0].flatten()
    time_stamp = np.arange(0, len(breath_signal))/sampling_rate

    assert (len(breath_signal) == len(ppg_signal)), "Mismatched length of breath signal and ppg signal."

    with h5py.File(output_path, 'w') as out_file:
        data_set = out_file.create_dataset('data', (len(ppg_signal),), dtype=[('signal', 'float32'), ('target', 'float32')])
        data_set['signal'] = ppg_signal
        data_set['target'] = breath_signal
        data_set.attrs['sampling_rate'] = sampling_rate
        data_set.attrs['signal_type'] = 'ppg'
        data_set.attrs['signal_sensor'] = 'max'
        data_set.attrs['target_type'] = 'thermistor'
        data_set.attrs['signal_sensor'] = 'green_thermistor'


def format_data_files(input_paths, output_paths, sampling_rate=200):
    """ Formats a list of files. Expects list of input paths and output paths with matching length. """
    assert (len(input_files) == len(output_files))
    for i_path, o_path in zip(input_paths, output_paths):
        format_data_file(i_path, o_path, sampling_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format collected max data to be of standard form.')
    parser.add_argument('input', type=str, help='Name of the max_thermistor raw signal file or directory of files to be formatted. For a directory extension of files is assumed to be ".csv".')
    parser.add_argument('output', type=str, help='Name of the file or directory to save the data to. Should use extension of ".h5".')
    parser.add_argument('--sampling_rate', type=int, default=200, help='Sampling rate at which the data was collected (Note that this should be consistent within one dataset).')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    sampling_rate = args.sampling_rate

    input_files=[]
    output_files=[]
    if os.path.isdir(input_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        elif not os.path.isdir(output_path):
            raise IOError("Specified output path must be a directory if input is a directory.")
        input_file_names = [os.path.basename(x) for x in glob.glob(os.path.join(input_path, '*.csv'))]
        input_files = [os.path.join(input_path, x) for x in input_file_names]
        output_files = [os.path.join(output_path, os.path.splitext(x)[0]+'.h5') for x in input_file_names]

    elif os.path.isfile(input_path):
        if os.path.isdir(output_path):
            raise IOError("Specified output path must be a file if input is a file.")
        input_files = [input_path]
        output_files = [output_path]
    else:
        raise IOError("Specified input path does not specify an existing file or directory.")

    format_data_files(input_files, output_files, sampling_rate)

