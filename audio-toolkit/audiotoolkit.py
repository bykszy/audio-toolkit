import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math


# import pysox


def add_echo(fs, x, offset_in_ms, alfa=0.4):
    offset = int(fs * offset_in_ms / 1000)  # ile próbek pominąć przed dodawaniem echa
    x1 = np.copy(x)
    for i in range(len(x) - offset):
        for j in range(x.shape[1]):
            x1[i + offset, j] += x1[i, j] * alfa
    return fs, x1


def add_noise(fs, x, snr=20, mean_noise=0, seed=-1):
    if seed != -1:
        np.random.seed(seed)

    rms_x = math.sqrt(np.mean(x ** 2))
    rms_noise = math.sqrt((rms_x ** 2) / (10 ** (snr / 10)))
    noise = np.random.normal(mean_noise, rms_noise, x.shape)
    y = np.copy(x)
    y = x + noise
    for i in range(len(x)):
        for j in range(x.shape[1]):
            if y[i, j] > 0.99:
                y[i, j] = 0.99
            if y[i, j] < -0.99:
                y[i, j] = -0.99
    return fs, y


def erase_freq(fs, x, seed=-1, f_range=100, where_to_begin=50):
    if seed != -1:
        np.random.seed(seed)

    if x.ndim == 2:
        xx = np.hsplit(x, 2)  # podział na kanały
        t = np.arange(0, 1, 1 / 80)
        x_fft_1 = np.fft.fft(xx[0])  # kanał lewy
        x_fft_2 = np.fft.fft(xx[1])  # kanał prawy
        print("kanał lewy i prawy fft")
        print(len(x_fft_1))
        print(len(x_fft_2))
        fig, axs = plt.subplots(2)
        fig.suptitle('erase freq widmo')
        axs[0].plot(np.abs(x_fft_1))
        axs[1].plot(np.abs(x_fft_2))
        plt.show()
        x1 = np.fft.ifft(x_fft_1)
        x2 = np.fft.ifft(x_fft_2)
        x_erased = np.concatenate((x1.real, x2.real), axis=1)
    else:
        x_fft_1 = np.fft.fft(x)
        x_fft_1_after = np.copy(x_fft_1)
        for i in range(f_range):
            x_fft_1_after[where_to_begin + i] = 0
        x_erased = np.fft.ifft(x_fft_1_after)
        fig, axs = plt.subplots(2)
        fig.suptitle('Spec aug widmo')
        axs[0].plot(np.abs(x_fft_1))
        axs[1].plot(np.abs(x_fft_1_after))
        plt.show()
        # print(np.sqrt(((x - x_spec) ** 2).mean()))
    return fs, x_erased


def specaug(fs, x, spec_or_mel, percent_frequency_erase, percent_time_erase, start_point_frequency=-1, start_point_time=-1, n_fft=2048, hop_length=512, n_mels=128):
    xx = np.hsplit(x, 2)  # podział na kanały
    if spec_or_mel == 0:
       # s1 = librosa.amplitude_to_db(np.abs(librosa.stft(xx[0].flatten())), ref=np.max)
       # s2 = librosa.amplitude_to_db(np.abs(librosa.stft(xx[1].flatten())), ref=np.max)
        s1 = abs(librosa.stft(xx[0].flatten()))
        s2 = abs(librosa.stft(xx[1].flatten()))
    else:
        s1 = librosa.feature.melspectrogram(xx[0].flatten(), sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        s2 = librosa.feature.melspectrogram(xx[1].flatten(), sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    if start_point_frequency == -1:
        for i in range(s1.shape[1]):
            for j in range(int((percent_frequency_erase * s1.shape[0])/100)):
                s1[int(100/percent_frequency_erase) * j, i] = 0
                s2[int(100/percent_frequency_erase) * j, i] = 0
    else:
        for i in range(s1.shape[1]):
            for j in range(int((percent_frequency_erase * s1.shape[0])/100)):
                s1[start_point_frequency + j, i] = 0
                s2[start_point_frequency + j, i] = 0

    if start_point_time == -1:
        for i in range(s1.shape[0]):
            for j in range(int((percent_time_erase * s1.shape[1])/100)):
                s1[i, int(100/percent_time_erase) * j] = 0
                s2[i, int(100/percent_time_erase) * j] = 0
    else:
        for i in range(s1.shape[0]):
            for j in range(int((percent_time_erase * s1.shape[1])/100)):
                s1[i, start_point_time + j] = 0
                s2[i, start_point_time + j] = 0
    s1_db = librosa.power_to_db(s1, ref=np.max)
    s2_db = librosa.power_to_db(s2, ref=np.max)
    # librosa.display.specshow(s1_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='mel')
    if spec_or_mel == 0:
        s2_db = librosa.power_to_db(s2, ref=np.max)
        print(s2)
        librosa.display.specshow(s2_db, sr=fs, y_axis='linear', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
    else:

        librosa.display.specshow(s2_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
    plt.show()
   # ss1 = librosa.feature.inverse.mel_to_stft(s1, sr=fs, n_fft=n_fft)
   # ss2 = librosa.feature.inverse.mel_to_stft(s2, sr=fs, n_fft=n_fft)
   # y1 = librosa.griffinlim(ss1)
   # y2 = librosa.griffinlim(ss2)
   # y_done = np.concatenate((y1.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
  #  fig, axs = plt.subplots(2)
   # fig.suptitle('Spec aug widmo')
    #axs[0].plot(xx[1])
   # axs[1].plot(y1)
   # plt.show()
    #return fs, y_done


def cut(fs, x, seconds=10, padding=0, time_offset=-1):  # padding = 0 uzupelnianie zerami, padding = 1
    if time_offset == -1 or time_offset > len(x):
        offset = np.random.randint(x.shape[0])
    else:
        offset = int(fs * time_offset)  # ilość próbek offsetu

    time = int(fs * seconds)  # ile próbek pobieramy

    if x.shape[1] == 2:
        x1 = np.zeros((time, 2))
        for i in range(time):
            for j in range(2):
                x1[i, j] = x[i + offset, j]
            if i + offset == len(x) - 1:
                break

    else:
        x1 = np.zeros(time)
        for i in range(time):
            x1[i] = x[i + time_offset]
        x1 = np.atleast_2d(x1)
        x1 = x1.T

    if offset + time > x.shape[0]:
        if padding == 1:
            for i in range(time + offset - x.shape[0]):
                for j in range(x1.shape[1]):
                    x1[i + x.shape[0] - offset - 1, j] = x[(i + offset) % x.shape[0], j]
        elif padding == 2:
            for i in range(time + offset - x.shape[0]):
                for j in range(x1.shape[1]):
                    # x1[i+offset+time-x.shape[0], j] = x[len(x)-i, j]
                    x1[i + x.shape[0] - offset - 1, j] = x[x.shape[0] - 1 - i % x.shape[0], j]
    return fs, x1


def reverse(fs, x):
    x1 = np.zeros((len(x), x.shape[1]))

    for i in range(len(x)):
        for j in range(x.shape[1]):
            x1[i, j] = x[len(x) - i - 1, j]

    return fs, x1


def mixup(fs1, x1, fs2, x2, alfa):
    fs1, x1 = cut(fs1, x1, 10)
    fs2, x2 = cut(fs2, x2, 10)
    if x1.shape[1] == 1 or x2.shape[1] == 1:
        x3 = np.zeros(x1.shape[0])
        for i in range(x1.shape[0]):
            x3[i] = x1[i, 0] * alfa + x2[i, 0] * (1 - alfa)
    else:
        x3 = x1 * alfa + x2 * (1 - alfa)
    return fs1, x1, x2, x3


def add_padding(x1, desired_len):
    zeros = np.zeros((desired_len - x1.shape[0], x1.shape[1]))
    y = np.concatenate((x1, zeros), axis=0)
    return y


def time_wrap(x1, percent):
    offset = int((percent / 100) * x1.shape[0])
    y = np.copy(x1)
    for i in range(x1.shape[0]):
        y[i] = x1[(i + offset) % x1.shape[0]]
    return y


