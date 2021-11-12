import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math
#import pysox



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
    rms_noise = math.sqrt(( rms_x ** 2) / (10 ** (snr / 10)))
    noise = np.random.normal(mean_noise, rms_noise, x.shape)
    y = x + noise

    return fs, y


def erase_freq(fs, x, seed=-1, f_range=100, where_to_begin=50):
    if seed != -1:
        np.random.seed(seed)

    if x.ndim == 2:
        xx = np.hsplit(x, 2)# podział na kanały
        t = np.arange(0, 1, 1 / 80)
        x_fft_1 = np.fft.fft(xx[0])#kanał lewy
        x_fft_2 = np.fft.fft(xx[1])#kanał prawy
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
        #print(np.sqrt(((x - x_spec) ** 2).mean()))
    return fs, x_erased


def specaug(fs, x, n_fft = 2048, hop_length = 512, n_mels = 128):
    xx = np.hsplit(x, 2)  # podział na kanały
    s1 = librosa.feature.melspectrogram(xx[0].flatten(), sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    s2 = librosa.feature.melspectrogram(xx[0].flatten(), sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    for i in range(s1.shape[1]):
        for j in range(28):
            s1[30 + j, i] = 0
            s2[30 + j, i] = 0
    s1_db = librosa.power_to_db(s1, ref=np.max)
    s2_db = librosa.power_to_db(s2, ref=np.max)
    #librosa.display.specshow(s1_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='mel')
    librosa.display.specshow(s2_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    ss1 = librosa.feature.inverse.mel_to_stft(s1, sr=fs, n_fft=n_fft)
    ss2 = librosa.feature.inverse.mel_to_stft(s2, sr=fs, n_fft=n_fft)
    y1 = librosa.griffinlim(ss1)
    y2 = librosa.griffinlim(ss2)
    y_done = np.concatenate((y1.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
    fig, axs = plt.subplots(2)
    fig.suptitle('Spec aug widmo')
    axs[0].plot(xx[1])
    axs[1].plot(y1)
    plt.show()
    return fs, y_done


def cut10s(fs, x, seconds=10, time_offset=-1):
    if seconds >= len(x) * fs:
        seconds = 10
    time = int(fs * seconds)  # ile próbek pobieramy
    if time_offset == -1 or time_offset > len(x):
        time_offset = np.random.randint(x.shape[0] - time)
    if x.shape[1] == 2:
        x1 = np.zeros((time, 2))
        for i in range(time):
            for j in range(2):
                x1[i, j] = x[i + time_offset, j]
    else:
        x1 = np.zeros(time)
        for i in range(time):
            x1[i] = x[i + time_offset]
        x1 = np.atleast_2d(x1)
        x1 = x1.T
    return fs, x1


def reverse(fs, x):
    x1 = np.zeros((len(x), x.shape[1]))

    for i in range(len(x)):
        for j in range(x.shape[1]):
            x1[i, j] = x[len(x) - i - 1, j]

    return fs, x1


def mixup(fs1, x1, fs2, x2, alfa):
    fs1, x1 = cut10s(fs1, x1)
    fs2, x2 = cut10s(fs2, x2)
    if x1.shape[1] == 1 or x2.shape[1] == 1:
        x3 = np.zeros(x1.shape[0])
        for i in range(x1.shape[0]):
            x3[i] = x1[i, 0] * alfa + x2[i, 0] * (1 - alfa)
    else:
        x3 = x1 * alfa + x2 * (1 - alfa)
    return fs1, x1, x2, x3

def resample(fs1, x1, final_fs):
    y = librosa.resample(x1, fs1, final_fs)
    return final_fs, y


def soxx():

    transformer = pysox.Transformer()  # create transformer
    transformer.trim(0, 15)  # trim the audio between 0 and 15 seconds
    transformer.build('wav-example.wav', 'sox.wav')  # create the output file