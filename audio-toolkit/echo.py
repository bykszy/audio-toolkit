import numpy as np
def add_echo(fs,x, offset_in_ms):
  offset=int(fs*offset_in_ms/1000)#ile próbek pominąć przed dodawaniem echa
  x1=np.copy(x)
  print(x1)
  for i in range(len(x)-offset):
    x1[i + offset, 0] += int(x1[i, 0] * 0.4)
    x1[i + offset, 1] += int(x1[i, 1] * 0.4)
  return fs, x1