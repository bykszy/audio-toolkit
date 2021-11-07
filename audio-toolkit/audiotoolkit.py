import numpy as np
import matplotlib.pyplot as plt
import librosa

def add_echo(fs, x, offset_in_ms):
    offset = int(fs * offset_in_ms / 1000)  # ile próbek pominąć przed dodawaniem echa
    x1 = np.copy(x)
    print(x1)
    for i in range(len(x) - offset):
        for j in range(x.shape[1]):
            x1[i + offset, j] += int(x1[i, j] * 0.4)
    return fs, x1


def add_noise(fs, x, target_noise_db=20, mean_noise=0, seed=-1):
    if seed != -1:
        np.random.seed(seed)

    x1 = np.copy(x)
    # noise = np.random.normal(0, 1, (len(x),x.ndim))

    target_noise_watts = 10 ** (target_noise_db / 10)

    noise_volts = np.random.normal(mean_noise, np.sqrt(target_noise_watts), (len(x), x.shape[1]))

    print(noise_volts)
    # Noise up the original signal (again) and plot
    y = x + noise_volts

    # for i in range(len(x)):
    # for j in range(x.ndim):
    # x1[i , j] += noise[i,j]
    return fs, y


def spec_aug(fs, x):
    xx = np.hsplit(x, 2)
    t = np.arange(0, 1, 1 / 80)
    x_fft_1 = np.fft.fft(xx[0])
    x_fft_2 = np.fft.fft(xx[1])
    fig, axs = plt.subplots(2)
    fig.suptitle('Spec aug widmo')
    axs[0].plot(np.abs(x_fft_1))
    axs[1].plot(np.abs(x_fft_2))
    plt.show()

    # for i in range(len(x)-offset):
    #  for j in range(x.shape[1]):
    #    x1[i + offset, j] += int(x1[i, j] * 0.4)


#  return fs, x1

def cut10s(fs, x, seconds=10, time_offset=-1):
    if seconds >= len(x) * fs:
        seconds = 10
    time = int(fs * seconds)  # ile próbek pobieramy
    if time_offset == -1 or time_offset > len(x):
        time_offset = np.random.randint(x.shape[0] - time)
    x1 = np.zeros((time, x.shape[1]))
    for i in range(time):
        x1[i] = x[i + time_offset]
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
    #np.zeros(x1.shape)
    x3 = x1 * alfa + x2 * (1 - alfa)

    return fs1, x1, x2, x3

def resample(fs1, x1, final_fs):
    y = librosa.resample(x1, fs1, final_fs)
    return final_fs, y