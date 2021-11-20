import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math
import os
from scipy.fftpack import fft, ifft


class AudioClip:
    def __init__(self, path=None, fs=24000):
        if path is None:
            self.path = str(os.path.dirname(os.path.abspath(__file__)))
            print(self.path)
        else:
            self.path = path

        self.audio = []

        if os.path.isdir(self.path):
            files = librosa.util.find_files(self.path, ext=['wav'])
            files = np.asarray(files)
            for y in files:
                print(str(y))
                data, sr = librosa.load(y, sr=fs, mono=False)
                data = np.atleast_2d(data).T
                self.audio.append([data, sr])

        if os.path.isfile(self.path):
            data, sr = librosa.load(self.path, sr=fs, mono=False)
            data = np.atleast_2d(data).T
            self.audio.append([data, sr])
            self.x_spec = []
        self.audio = np.array(self.audio, dtype=object) ####### self.audio format -> [different song (0,1,2,3,...), data(0) or fs(1) ][data samples, channel]

    def add_echo(self, offset_in_ms, alfa=0.4):
        for a in range(len(self.audio)):
            offset = int(self.audio[a, 1] * offset_in_ms / 1000)  # ile próbek pominąć przed dodawaniem echa
            x1 = np.copy(self.audio[a, 0])
            for i in range(len(self.audio[a, 0]) - offset):
                for j in range(self.audio[a, 0].shape[1]):
                    self.audio[a, 0][i + offset, j] += self.audio[a, 0][i, j] * alfa
            return self

    def add_noise(self, snr=20, mean_noise=0, seed=-1):
        for a in range(len(self.audio)):
            if seed != -1:
                np.random.seed(seed)

            rms_x = math.sqrt(np.mean(self.audio[a, 0] ** 2))
            rms_noise = math.sqrt((rms_x ** 2) / (10 ** (snr / 10)))
            noise = np.random.normal(mean_noise, rms_noise, self.audio[a, 0].shape)
            y = np.copy(self.audio[a, 0])
            self.audio[a, 0] = self.audio[a, 0] + noise
            for i in range(len(self.audio[a, 0])):
                for j in range(self.audio[a, 0].shape[1]):
                    if self.audio[a, 0][i, j] > 0.99:
                        self.audio[a, 0][i, j] = 0.99
                    if self.audio[a, 0][i, j] < -0.99:
                        self.audio[a, 0][i, j] = -0.99
        return self

    def erase_freq(self, f_range=100, where_to_begin=50, seed=-1):
        for a in range(len(self.audio)):
            if seed != -1:
                np.random.seed(seed)

            if self.audio[a, 0].ndim == 2:
                xx = np.hsplit(self.audio[a, 0], 2)  # podział na kanały
                t = np.arange(0, 1, 1 / 80)
                a1 = self.audio[a, 0].T[0]
                a2 = self.audio[a, 0].T[1]# this is a two channel soundtrack, I get the first track
                #b = [(ele / 2 ** 8.) * 2 - 1 for ele in aa]  # this is 8-bit track, b is now normalized on [-1,1)
                c1 = fft(a1)
                c2 = fft(a2) # create a list of complex number
                d = len(c1) / 2
                for i in range(f_range):
                    c1[where_to_begin + i] = 0
                    c2[where_to_begin + i] = 0
                    c1[self.audio[a, 0].shape[0] - where_to_begin - i] = 0
                    c2[self.audio[a, 0].shape[0] - where_to_begin - i] = 0
                fig, axs = plt.subplots(2)
                fig.suptitle('erase freq widmo')
                axs[0].plot(abs(c1))
                axs[1].plot(abs(c2))
                # axs[0].plot(abs(c1[:int(d-1)]))
                # axs[1].plot(abs(c2[:int(d-1)]))
                plt.show()
                x1 = ifft(c1)
                x2 = ifft(c2)
                x1 = np.atleast_2d(x1.real).T
                x2 = np.atleast_2d(x2.real).T
                self.audio[a, 0] = np.concatenate((x1.real, x2.real), axis=1)
            else:
                x_fft_1 = np.fft.fft(self.audio[a, 0], axis=0)
                x_fft_1_after = np.copy(x_fft_1)
                for i in range(f_range):
                    x_fft_1_after[where_to_begin + i] = 0
                self.audio[a, 0] = np.fft.ifft(x_fft_1_after)
                fig, axs = plt.subplots(2)
                fig.suptitle('Spec aug widmo')
                axs[0].plot(np.abs(x_fft_1))
                axs[1].plot(np.abs(x_fft_1_after))
                plt.show()
                self.audio[a, 0] = np.fft.ifft(x_fft_1_after)
        return self

    def specaug(self, spec_or_mel, percent_frequency_erase, percent_time_erase, start_point_frequency=-1,
                start_point_time=-1, n_fft=2048, hop_length=512, n_mels=128):
        for a in range(len(self.audio)):
            xx = np.hsplit(self.audio[a, 0], 2)  # podział na kanały
            if spec_or_mel == 0:
                # s1 = librosa.amplitude_to_db(np.abs(librosa.stft(xx[0].flatten())), ref=np.max)
                # s2 = librosa.amplitude_to_db(np.abs(librosa.stft(xx[1].flatten())), ref=np.max)
                s1 = abs(librosa.stft(xx[0].flatten()))
                s2 = abs(librosa.stft(xx[1].flatten()))
            else:
                s1 = librosa.feature.melspectrogram(xx[0].flatten(), sr=self.audio[a, 1], n_fft=n_fft, hop_length=hop_length,
                                                    n_mels=n_mels)
                s2 = librosa.feature.melspectrogram(xx[1].flatten(), sr=self.audio[a, 1], n_fft=n_fft, hop_length=hop_length,
                                                    n_mels=n_mels)
            if start_point_frequency == -1:
                for i in range(s1.shape[1]):
                    for j in range(int((percent_frequency_erase * s1.shape[0]) / 100)):
                        s1[int(100 / percent_frequency_erase) * j, i] = 0
                        s2[int(100 / percent_frequency_erase) * j, i] = 0
            else:
                for i in range(s1.shape[1]):
                    for j in range(int((percent_frequency_erase * s1.shape[0]) / 100)):
                        s1[start_point_frequency + j, i] = 0
                        s2[start_point_frequency + j, i] = 0

            if start_point_time == -1:
                for i in range(s1.shape[0]):
                    for j in range(int((percent_time_erase * s1.shape[1]) / 100)):
                        s1[i, int(100 / percent_time_erase) * j] = 0
                        s2[i, int(100 / percent_time_erase) * j] = 0
            else:
                for i in range(s1.shape[0]):
                    for j in range(int((percent_time_erase * s1.shape[1]) / 100)):
                        s1[i, start_point_time + j] = 0
                        s2[i, start_point_time + j] = 0
            s1_db = librosa.power_to_db(s1, ref=np.max)
            s2_db = librosa.power_to_db(s2, ref=np.max)
            # librosa.display.specshow(s1_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='mel')
            if spec_or_mel == 0:
                librosa.display.specshow(s2_db, sr=self.audio[a, 1], y_axis='linear', x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                self.x_spec[a] = np.concatenate((s1_db, s2_db), axis=1)
            else:
                librosa.display.specshow(s2_db, sr=self.audio[a, 1], hop_length=hop_length, x_axis='time', y_axis='mel')
                plt.colorbar(format='%+2.0f dB')
                self.x_spec[a] = np.concatenate((s1_db, s2_db), axis=1)
            plt.show()
        return self

    def cut(self, seconds=10, padding=0, time_offset=-1):  # padding = 0 uzupelnianie zerami, padding = 1
        for a in range(len(self.audio)):
            if time_offset == -1 or time_offset > len(self.audio[a, 0]):
                offset = np.random.randint(self.audio[a, 0].shape[0])
            else:
                offset = int(self.audio[a, 1] * time_offset)  # ilość próbek offsetu
            time = int(self.audio[a, 1] * seconds)  # ile próbek pobieramy

            if self.audio[a, 0].shape[1] == 2:
                self.audio[a, 0] = self.audio[a, 0][0:time]
                for i in range(time):
                    for j in range(2):
                        self.audio[a, 0][i, j] = self.audio[a, 0][i + offset, j]
                    if i + offset == len(self.audio[a, 0]) - 1:
                        break

            else:
                self.audio[a, 0] = self.audio[a, 0][0:time]
                for i in range(time):
                    self.audio[a, 0][i] = self.audio[a, 0][i + time_offset]

            if offset + time > self.audio[a, 0].shape[0]:
                if padding == 1:
                    for i in range(time + offset - self.audio[a, 0].shape[0]):
                        for j in range(self.audio[a, 0].shape[1]):
                            self.audio[a, 0][i + self.audio[a, 0].shape[0] - offset - 1, j] = self.audio[a, 0][
                                (i + offset) % self.audio[a, 0].shape[0], j]
                elif padding == 2:
                    for i in range(time + offset - self.audio[a, 0].shape[0]):
                        for j in range(self.audio[a, 0].shape[1]):
                            self.audio[a, 0][i + self.audio[a, 0].shape[0] - offset - 1, j] = self.audio[a, 0][
                                self.audio[a, 0].shape[0] - 1 - i % self.audio[a, 0].shape[0], j]
        return self

    def reverse(self):
        for a in range(len(self.audio)):
            x1 = np.copy(self.audio[a, 0])
            for i in range(len(self.audio[a, 0])):
                for j in range(self.audio[a, 0].shape[1]):
                    self.audio[a, 0][i, j] = x1[len(self.audio[a, 0]) - i - 1, j]
        return self

    def mixup(self, fs1, x1, fs2, x2, alfa):
        #  fs1, x1 = self.cut(10) #####################################
        #  fs2, x2 = self.cut(10) #####################################
        if x1.shape[1] == 1 or x2.shape[1] == 1:
            x3 = np.zeros(x1.shape[0])
            for i in range(x1.shape[0]):
                x3[i] = x1[i, 0] * alfa + x2[i, 0] * (1 - alfa)
        else:
            x3 = x1 * alfa + x2 * (1 - alfa)
        return fs1, x1, x2, x3

    def add_padding(self, desired_len):
        for a in range(len(self.audio)):
            zeros = np.zeros((desired_len - self.audio[a, 0].shape[0], self.audio[a, 0].shape[1]))
            self.audio[a, 0] = np.concatenate((self.audio[a, 0], zeros), axis=0)
        return self

    def time_wrap(self, percent):
        for a in range(len(self.audio)):
            offset = int((percent / 100) * self.audio[a, 0].shape[0])
            # y = np.copy(self.x)
            for i in range(self.audio[a, 0].shape[0]):
                self.audio[a, 0][i] = self.audio[a, 0][(i + offset) % self.audio[a, 0].shape[0]]
        return self

    def get_audio(self):
        return self.audio

    #def get_x_fs(self):
     #   for a in range(len(self.audio)):
      #      return self.x, self.audio[a, 1]

    def get_spectograms(self):
        return self.x_spec

    #def get_fs(self):
     #   return self.fs
