import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from echo import add_echo
from reverse import reverse
from cut10s import cut10s
from scipy.io.wavfile import read, write

#fs, x = read('wav-example.wav')
fs, x = read('record.wav')
print(fs)

t = np.arange(0, len(x)) / fs

#plt.figure(figsize=(20, 7))
#plt.plot(t, x, label='Original signal')


fs2, x2 = reverse(fs, x)
fs1, x1 = add_echo(fs, x, 100)
fs3, x3 = cut10s(fs, x,1)

fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(np.arange(0, len(x)) / fs, x)
axs[1].plot(np.arange(0, len(x1)) / fs1, x1)
axs[2].plot(np.arange(0, len(x2)) / fs2, x2)
axs[3].plot(np.arange(0, len(x3)) / fs3, x3)
plt.show()
write("record_echo.wav", fs1, x1.astype(np.int16))
write("record_reversed.wav", fs2, x2.astype(np.int16))
write("record_cut1s.wav", fs3, x3.astype(np.int16))


