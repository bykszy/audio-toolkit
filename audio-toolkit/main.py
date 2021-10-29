import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from audiotoolkit import *
from scipy.io.wavfile import read, write

# fs, x = read('wav-example.wav')
fs, x_talk = read('record.wav')
fs_toilet, x_toilet = read('toilet_flush.wav')
print(fs)
print(fs_toilet)
t = np.arange(0, len(x_talk)) / fs

spec_aug(fs, x_talk)
plt.figure(figsize=(20, 7))
plt.plot(t, x_talk, label='Original signal')

fs_echo, x_echo = add_echo(fs, x_talk, 100)
fs_reverse, x_reverse = reverse(fs, x_talk)
fs_cut10s, x_cut10s = cut10s(fs, x_talk, 10)
fs_noise, x_noise = add_noise(fs, x_talk)
fs_mixup, x_mixup_1, x_mixup_2, x_mixup = mixup(fs, x_talk, fs_toilet, x_toilet, 0.8)
fig, axs = plt.subplots(3)
fig.suptitle('1, 2, Mixup')
axs[0].plot(np.arange(0, len(x_mixup_1)) / fs_mixup, x_mixup_1)
axs[1].plot(np.arange(0, len(x_mixup_2)) / fs_mixup, x_mixup_2)
axs[2].plot(np.arange(0, len(x_mixup)) / fs_mixup, x_mixup)

plt.show()

fig, axs = plt.subplots(5)
fig.suptitle('1-Normal, 2-echo, 3-reverse, 4-cut10s, 5-noise')
axs[0].plot(np.arange(0, len(x_talk)) / fs, x_talk)
axs[1].plot(np.arange(0, len(x_echo)) / fs_echo, x_echo)
axs[2].plot(np.arange(0, len(x_reverse)) / fs_reverse, x_reverse)
axs[3].plot(np.arange(0, len(x_cut10s)) / fs_cut10s, x_cut10s)
axs[4].plot(np.arange(0, len(x_noise)) / fs_noise, x_noise)
plt.show()
write("record_echo.wav", fs_echo, x_echo.astype(np.int16))
write("record_reverse.wav", fs_reverse, x_reverse.astype(np.int16))
write("record_cut10s.wav", fs_cut10s, x_cut10s.astype(np.int16))
write("record_noise.wav", fs_noise, x_noise.astype(np.int16))

write("record_mixup_speech_toilet.wav", fs_mixup, x_mixup.astype(np.int16))
