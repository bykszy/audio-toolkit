import numpy as np
def reverse(fs,x):

  x1=np.zeros((len(x), x.ndim))

  for i in range(len(x)):
    for j in range(x.ndim):
      x1[i, j] = x[len(x)-i-1, j]

  return fs, x1