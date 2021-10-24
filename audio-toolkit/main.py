import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.wavfile import read, write


fs, x = read('wav-example.wav')
print(fs)

t = np.arange(0, len(x))/fs


plt.figure(figsize=(20, 7))
plt.plot(t, x, label='Original signal')
plt.show()

write("savetestexample1.wav", fs, x.astype(np.int16))

