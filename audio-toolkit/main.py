import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io.wavfile import read, write
from playsound import playsound

fs, x = read('wav-example.wav')
t = np.arange(0, len(x))/fs
print(len(t))
plt.figure(figsize=(20, 7))
plt.plot(t, x, label='Original signal')
plt.show()

playsound('audio.mp3')