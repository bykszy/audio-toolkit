import numpy as np
def add_echo(fs,x, offset_in_ms):
  offset=int(fs*offset_in_ms/1000)#ile próbek pominąć przed dodawaniem echa
  x1=np.copy(x)
  print(x1)
  for i in range(len(x)-offset):
    for j in range(x.ndim):
      x1[i + offset, j] += int(x1[i, j] * 0.4)
  return fs, x1