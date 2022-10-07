import okodm_class
import time
import sys

h=okodm_class.open("MMDM 37ch,15mm","USB DAC 40ch, 12bit",["D40V2X02"])

if h==0:
    sys.exit("Error opening OKODM device: "+str(okodm_class.lasterror()))
    
n=okodm_class.chan_n(h)
stage=1   
   
try:
    while True:
        if not okodm_class.set(h, ([1] if stage else [-1])*n):
            sys.exit("Error writing to OKODM device: "+okodm_class.lasterror())
        time.sleep(1)
        stage^=1
        
except KeyboardInterrupt:
    pass        
  
okodm_class.close(h)  

