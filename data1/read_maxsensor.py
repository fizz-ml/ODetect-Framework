import serial
import time
import sys
path = sys.argv[1]
ser = serial.Serial('/dev/ttyACM0', 115200) # Establish the connection on a specific port
with open(path, 'w') as f:
   while(True):
       try:
           val = int(ser.readline())
           thing = "{}, {}\n".format(time.time(), val)
           f.write(thing)
           #print(thing)
       except KeyboardInterrupt:
           break
       except:
           #fuck all
           pass
