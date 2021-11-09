import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from audiotoolkit import *
from scipy.io.wavfile import read, write
import librosa

fs = 48000

# fs, x = read('wav-example.wav')
x, fs_x = librosa.load('wav-example.wav', sr=fs, mono=False)

x_talk, fs_talk = librosa.load('record.wav', sr=fs, mono=False)
x_toilet, fs_toilet = librosa.load('toilet_flush.wav', sr=fs)
x = np.atleast_2d(x)
x = np.array(x).T
x_talk = np.atleast_2d(x_talk)
x_talk = np.array(x_talk).T
x_toilet = np.atleast_2d(x_toilet)
x_toilet = np.array(x_toilet).T
fs_mel, x_mel = mel_display(fs_x, x)
write("record_example_mel.wav", fs_mel, x_mel.astype(np.float32))

"""
print(x_talk.shape)
print(x.shape)
print(x_toilet.shape)
print(x_toilet)
t1 = np.arange(0, len(x)) / fs_x
t2 = np.arange(0, len(x_talk)) / fs_talk
t3 = np.arange(0, len(x_toilet)) / fs_toilet
# spec_aug(fs, x_talk)
# fs_x_spec, x_spec = erase_freq(fs_x, x)
# erase_freq(fs_toilet, x_toilet)
# erase_freq(fs_resampled, x_resampled)
fig, axs = plt.subplots(3)
fig.suptitle('startowe')
axs[0].plot(t1, x)
axs[1].plot(t2, x_talk)
axs[2].plot(t3, x_toilet)
plt.show()
# plt.plot(t, x_talk, label='Original signal')
write("test1.wav", fs, x.astype(np.int16))


fs_echo, x_echo = add_echo(fs, x_talk, 100)

fs_reverse, x_reverse = reverse(fs, x_talk)

fs_cut10s, x_cut10s = cut10s(fs, x_talk, 10)

fs_noise, x_noise = add_noise(fs, x_talk, 10)

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
write("record_echo.wav", fs_echo, x_echo.astype(np.float32))
write("record_reverse.wav", fs_reverse, x_reverse.astype(np.float32))
write("record_cut10s.wav", fs_cut10s, x_cut10s.astype(np.float32))
write("record_noise.wav", fs_noise, x_noise.astype(np.float32))
write("record_mixup_speech_toilet.wav", fs_mixup, x_mixup.astype(np.float32))
"""