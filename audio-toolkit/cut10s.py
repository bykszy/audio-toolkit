import numpy as np
def cut10s(fs,x,seconds=10,time_offset=5):
  if(seconds>=len(x)*fs):
    seconds=10
  time=int(fs*seconds)#ile próbek pobieramy
  offset=int(fs*time_offset)
  x1=np.zeros((time, x.ndim))
  if(offset>len(x)):
    offset=np.random.randint(0,len(x)-offset)
  for i in range(time):
    x1[i]=x[i+offset]
  return fs, x1