import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.integrate import cumtrapz
# from session import *
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from custom_collates import CustomCollate
from config import option
from utils.session import Sessions
from utils.util import seed

opti = option()
opti.initialize()
opti.parser.load = True
opti.parser.input_filename = 'SPAR.npy'
opti.parser.metrics = 'NA'
opt = opti.process()
session = Sessions(opt)
# session
har_data = session.output_data()
subject = har_data['subject']
X = har_data['X']
y = har_data['y']

print(dict(har_data).keys())
print()
dt = 1.0 / 50.0
print(set(y))


def get_index_by_subject(subject_id, subject, X, y):
    res = []
    for i in range(len(subject)):
        if subject[i] == subject_id:
            res.append(i)
    return res


res = get_index_by_subject(1, subject, X, y)
X_temp = []
y_temp = []
side = []
for i in res:
    if har_data['side'][i] == 1.0:
        X_temp.append(X[i])
        y_temp.append(y[i])
        side.append(har_data['side'][i])
print(len(y_temp))


def shift(x):
    #     plt.plot(abs(np.fft.rfft(x - np.average(x))))
    #     plt.show()
    return x - np.average(x)


def visual_by_index(Xs, y, n=7):
    titles = list(set(har_data['y']))
    fig, axs = plt.subplots(4, n, sharey='row', figsize=(15, 10))
    res_vel = []
    for index, (acc, y_index) in enumerate(zip(Xs[0:n], y[0:n])):
        a0 = acc[:, 0]
        a1 = acc[:, 1]
        a2 = acc[:, 2]
        #         velocity = scipy.integrate.simps(each, x=np.arange(len(each)))
        #         print(velocity)
        axs[0, y_index].set_title(titles[index])
        axs[0, y_index].plot(acc)

        freq = np.fft.rfftfreq(len(a0), d=1 / 50)

        f0 = abs(np.fft.rfft(a0))
        f1 = abs(np.fft.rfft(a1))
        f2 = abs(np.fft.rfft(a2))

        axs[1, y_index].plot(freq, f0)
        axs[2, y_index].plot(freq, f1)
        axs[3, y_index].plot(freq, f2)
    #         res_vel.append()
    plt.show()
    return res_vel


def fourierr(y, xs=2, f=50.0, N=None):
    # sample spacing
    if N is None: N = len(y)
    T = 1.0 / f
    x = np.linspace(0.0, N * T, N)

    # y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    yf = torch.fft.rfft(torch.tensor(y))
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    size = len(np.linspace(0.0, xs, N // (2 * int(f / 2 / xs))))
    return xf[:size], 2.0 / N * np.abs(yf[:N // 2])[:size]


def hpf(data, gthreshold=1, allenuate_rate=.1):
    freq = np.fft.rfftfreq(len(data), d=1 / 50)
    fft_x = np.fft.rfft(data)

    return np.fft.irfft(
        np.where((freq > gthreshold), fft_x * allenuate_rate, fft_x),
        n=data.shape[0]
    )


def lpf(data, lthreshold=.3, allenuate_rate=.1):
    freq = np.fft.rfftfreq(len(data), d=1 / 50)
    fft_x = np.fft.rfft(data)

    return np.fft.irfft(
        np.where((freq < lthreshold), fft_x * allenuate_rate, fft_x),
        n=data.shape[0]
    )


r = hpf(X_temp[0][:, 0])
# 50 / len(r)
freq = np.fft.rfftfreq(len(r), d=1 / 50)
plt.plot(freq, abs(np.fft.rfft(r)))
plt.show()
xf, yf = fourierr(r, xs=25)
plt.plot(xf, yf)

visual_by_index(X_temp, y_temp)


def prior_acc(X, y, n=7):
    titles = har_data['y_labels']
    res_vel = []
    # max
    res_fourier = np.ones((3, 7))
    res_fourier_counter = np.zeros((3, 7))
    res_fourier *= -1000.0
    #     length = int(np.average([len(each_x) for each_x in X]))
    for index, (each_x, each_y) in enumerate(zip(X, y)):

        for acc_index in range(3):
            a = each_x[:, acc_index]
            freq, f = fourierr(a, xs=25)
            maxf = max(f)
            res_fourier_counter[acc_index, each_y] += 1
            if maxf > res_fourier[acc_index, each_y]:
                res_fourier[acc_index, each_y] = maxf
    #     print(res_fourier_counter)
    #     res_fourier /= res_fourier_counter
    return res_fourier
