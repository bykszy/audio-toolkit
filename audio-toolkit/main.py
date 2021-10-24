import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from echo import add_echo
from scipy.io.wavfile import read, write


fs, x = read('wav-example.wav')
print(fs)

t = np.arange(0, len(x))/fs


plt.figure(figsize=(20, 7))
plt.plot(t, x, label='Original signal')
plt.show()

fs1, x1 = add_echo(fs, x, 100)

write("echotest2.wav", fs1, x1.astype(np.int16))

