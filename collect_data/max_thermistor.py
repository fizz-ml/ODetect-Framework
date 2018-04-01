"""
This script was written to collect simultaneous data from the max and thermistor sensors.
It is to be used in combination with max_thermistor.ino which is an Arduino script.
The script makes use of the pyserial module.
"""

import argparse
import serial
import time

def collect_data(serial_port='/dev/ttyACM0', baud_rate=115200):
    ser = serial.Serial(serial_port, baud_rate)
    data_list = []
    while(True):
       try:
           val = ser.readline().strip().decode("utf-8")

           # Ignore initial message
           if "nitial" in val:
               continue

           # Add timestamp to output
           csv_row = "{}, {}\n".format(time.time(), val)

           data_list.append(csv_row)
           print(csv_row)
       except KeyboardInterrupt:
           break
       except:
           # Ignore bad lines
           pass
    return data_list

def save_data(path, data_list):
    with open(path, 'w') as f:
        for num, i in enumerate(data_list):
            # ignore the first 10 lines as it can start out zeroed
            if num > 10:
                f.write(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect simultaneous max and thermistor readings. Assumes serial port is receiving data as formatted in the Arduino script.')
    parser.add_argument('save_path', type=str, help='Name of the file to save the data to.')
    parser.add_argument('--serial_port_path', type=str, default='/dev/ttyACM0', help='File path to the serial port on which Arduino is communicating.')
    # parser.add_argument('--sampling_rate', type=int, default=200, help='Sampling rate at which the data is collected (Note that this should be consistent within one dataset).')
    parser.add_argument('--baud_rate', type=int, default=115200, help='Baud rate for the serial communication (Not tested outside default values).')
    args = parser.parse_args()

    save_path = args.save_path
    serial_port_path = args.serial_port_path
    # sampling_rate = args.sampling_rate
    baud_rate = args.baud_rate

    data_list = collect_data(serial_port=serial_port_path, baud_rate=baud_rate)
    save_data(save_path, data_list)
