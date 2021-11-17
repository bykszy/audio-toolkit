import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math


class AudioClip:
    def __init__(self, path, fs=24000):
        self.path = path
         #= fs
        self.x, self.fs = librosa.load(self.path, sr=fs, mono=False)
        self.x = np.atleast_2d(self.x).T
        self.x_spec = []




    def add_echo(self, offset_in_ms, alfa=0.4):
        offset = int(self.fs * offset_in_ms / 1000)  # ile próbek pominąć przed dodawaniem echa
        x1 = np.copy(self.x)
        for i in range(len(self.x) - offset):
            for j in range(self.x.shape[1]):
                self.x[i + offset, j] += self.x[i, j] * alfa
        return self


    def add_noise(self, snr=20, mean_noise=0, seed=-1):
        if seed != -1:
            np.random.seed(seed)

        rms_x = math.sqrt(np.mean(self.x ** 2))
        rms_noise = math.sqrt((rms_x ** 2) / (10 ** (snr / 10)))
        noise = np.random.normal(mean_noise, rms_noise, self.x.shape)
        y = np.copy(self.x)
        self.x = self.x + noise
        for i in range(len(self.x)):
            for j in range(self.x.shape[1]):
                if self.x[i, j] > 0.99:
                    self.x[i, j] = 0.99
                if self.x[i, j] < -0.99:
                    self.x[i, j] = -0.99
        return self


    def erase_freq(self, seed=-1, f_range=100, where_to_begin=50):
        if seed != -1:
            np.random.seed(seed)

        if self.x.ndim == 2:
            xx = np.hsplit(self.x, 2)  # podział na kanały
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
            self.x = np.concatenate((x1.real, x2.real), axis=1)
        else:
            x_fft_1 = np.fft.fft(self.x)
            x_fft_1_after = np.copy(x_fft_1)
            for i in range(f_range):
                x_fft_1_after[where_to_begin + i] = 0
            self.x = np.fft.ifft(x_fft_1_after)
            fig, axs = plt.subplots(2)
            fig.suptitle('Spec aug widmo')
            axs[0].plot(np.abs(x_fft_1))
            axs[1].plot(np.abs(x_fft_1_after))
            plt.show()
            # print(np.sqrt(((x - x_spec) ** 2).mean()))
        return self


    def specaug(self, spec_or_mel, percent_frequency_erase, percent_time_erase, start_point_frequency=-1,
                start_point_time=-1, n_fft=2048, hop_length=512, n_mels=128):
        xx = np.hsplit(self.x, 2)  # podział na kanały
        if spec_or_mel == 0:
            # s1 = librosa.amplitude_to_db(np.abs(librosa.stft(xx[0].flatten())), ref=np.max)
            # s2 = librosa.amplitude_to_db(np.abs(librosa.stft(xx[1].flatten())), ref=np.max)
            s1 = abs(librosa.stft(xx[0].flatten()))
            s2 = abs(librosa.stft(xx[1].flatten()))
        else:
            s1 = librosa.feature.melspectrogram(xx[0].flatten(), sr=self.fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            s2 = librosa.feature.melspectrogram(xx[1].flatten(), sr=self.fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
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
            librosa.display.specshow(s2_db, sr=self.fs, y_axis='linear', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            self.x_spec = np.concatenate((s1_db, s2_db), axis=1)
        else:
            librosa.display.specshow(s2_db, sr=self.fs, hop_length=hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            self.x_spec = np.concatenate((s1_db, s2_db), axis=1)
        plt.show()
        return self


    def cut(self, seconds=10, padding=0, time_offset=-1):  # padding = 0 uzupelnianie zerami, padding = 1
        if time_offset == -1 or time_offset > len(self.x):
            offset = np.random.randint(self.x.shape[0])
        else:
            offset = int(self.fs * time_offset)  # ilość próbek offsetu
        print("fs :" +str(self.fs))
        time = int(self.fs * seconds)  # ile próbek pobieramy

        if self.x.shape[1] == 2:
            #print("fs :" + str(self.fs))
            self.x = self.x[0:time]
            for i in range(time):
                for j in range(2):
                    #x1[i, j] = self.x[i + offset, j]
                    self.x[i, j] = self.x[i + offset, j]
                if i + offset == len(self.x) - 1:
                    break

        else:
            #self.x.resize((time, 1))
            self.x = self.x[0:time]
            for i in range(time):
                #x1[i] = self.x[i + time_offset]
                self.x[i] = self.x[i + time_offset]
           # x1 = np.atleast_2d(x1)
            #x1 = x1.T

        if offset + time > self.x.shape[0]:
            if padding == 1:
                for i in range(time + offset - self.x.shape[0]):
                    for j in range(self.x.shape[1]):
                        #x1[i + self.x.shape[0] - offset - 1, j] = self.x[(i + offset) % self.x.shape[0], j]
                        self.x[i + self.x.shape[0] - offset - 1, j] = self.x[(i + offset) % self.x.shape[0], j]
            elif padding == 2:
                for i in range(time + offset - self.x.shape[0]):
                    for j in range(x1.shape[1]):
                        #x1[i + self.x.shape[0] - offset - 1, j] = self.x[self.x.shape[0] - 1 - i % self.x.shape[0], j]
                        self.x[i + self.x.shape[0] - offset - 1, j] = self.x[self.x.shape[0] - 1 - i % self.x.shape[0], j]
        return self


    def reverse(self):
        x1 = np.copy(self.x)
        for i in range(len(self.x)):
            for j in range(self.x.shape[1]):

                self.x[i, j] = x1[len(self.x) - i - 1, j]

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
        zeros = np.zeros((desired_len - self.x.shape[0], self.x.shape[1]))
        self.x = np.concatenate((self.x, zeros), axis=0)
        return self


    def time_wrap(self, percent):
        offset = int((percent / 100) * self.x.shape[0])
        #y = np.copy(self.x)
        for i in range(self.x.shape[0]):
            self.x[i] = self.x[(i + offset) % self.x.shape[0]]
        return self

    def get_x_fs(self):
        return self.x, self.fs

    def get_xspec(self):
        return self.x_spec

    def get_fs(self):
        return self.fs
