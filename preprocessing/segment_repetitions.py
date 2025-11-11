import csv
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.signal import argrelextrema, find_peaks

NUM_OF_REPETITION = 10
NUM_OF_SPLIT = NUM_OF_REPETITION - 1

input_directory = './data_ROM/'
output_directory = './segment_sessions_one_repetition_data_'
one_repetition_data_recording = './segment_sessions_one_repetition_data_'
figure_directory = './segment_sessions_one_repetition_figures_'
WEIGHT = [1, .2, 1.5]  #[1, .2, 1.5]


# plt.ion()
def standardize(train_data):
    train_data = torch.tensor(np.array(train_data))
    means = train_data.mean(0, keepdim=True)
    stds = train_data.std(0, keepdim=True)
    normalized_data = (train_data - means) / stds
    return normalized_data


def update_foldernames(filename):
    global output_directory
    global one_repetition_data_recording
    global figure_directory
    exercise = filename.split('_')[1]
    assert exercise.startswith('E') and len(
        exercise
    ) > 1, "Name labeling in exercise is incorrect:" + str(exercise)
    output_directory += exercise + "/"
    one_repetition_data_recording += exercise + ".txt"
    figure_directory += exercise + "/"
    print(output_directory)
    print(one_repetition_data_recording)
    print(figure_directory)

    # check exists
    CHECK_FOLDER = os.path.isdir(output_directory)
    if not CHECK_FOLDER:
        os.makedirs(output_directory)
    CHECK_FOLDER = os.path.isdir(figure_directory)
    if not CHECK_FOLDER:
        os.makedirs(figure_directory)
    CHECK_FOLDER = os.path.exists(one_repetition_data_recording)
    if not CHECK_FOLDER:
        f = open(one_repetition_data_recording, 'w')
        f.close()


def reset_foldernames():
    global output_directory
    global one_repetition_data_recording
    global figure_directory
    output_directory = './segment_sessions_one_repetition_data_'
    one_repetition_data_recording = './segment_sessions_one_repetition_data_'
    figure_directory = './segment_sessions_one_repetition_figures_'


def cal_energy(size, x, weighted=[1, 1, 1]):
    new_A = []
    for each in x:
        # print(each.shape)
        s_x = each[0] * weighted[0]
        s_y = each[1] * weighted[1]
        s_z = each[2] * weighted[2]
        a = sum([abs(inf) for inf in [s_x, s_y, s_z]]) / len(each)
        new_A.append(a)
    energy = []
    print("length is ", len(new_A))
    for i in range(len(new_A)):

        if i - size >= 0 and i + size <= len(new_A):
            temp = new_A[i - size:i + size]
        elif i - size >= 0 and i + size > len(new_A):
            length = size - (len(new_A) - i)
            temp = new_A[i - size:i] + new_A[i:] + new_A[0:length]
        else:

            length = len(new_A) - size + i
            temp = new_A[length:] + new_A[0:i] + new_A[i:i + size
                                                       ]  # A[length:] +

        energy.append(
            sum([np.sqrt(abs(ele)) for ele in temp]) / len(temp)
        )  # originally was \ len(temp)
    return standardize(energy)
    # return energy


def fourier(y, N, T=1. / 50.):
    # sample spacing
    T = 1.0 / 50.0
    x = np.linspace(0.0, N * T, N)

    # y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    yf = torch.fft.fft(torch.tensor(y))
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    return xf, 2.0 / N * np.abs(yf[:N // 2])


for dirname, dirnames, filenames in os.walk(input_directory):
    for filename in filenames:
        if filename == '.DS_Store':
            continue
        looks_good = False
        update_foldernames(filename)
        f = open(one_repetition_data_recording, 'a')
        while not looks_good:
            # if not filename.startswith("S5"):
            #     continue
            session = os.path.join(dirname, filename)
            print('-------------', 'this file is', session, '-------------')
            data = pd.read_csv(session, delimiter=',', index_col=False)
            HEADER = list(data.columns)
            acc = data.columns[1:4]

            x_pd = data[acc]

            N = len(data[data.columns[1:7]].to_numpy())
            print(N)
            # fig, ax = plt.subplots()
            # for each in range(1, 7):
            #     # sample spacing
            #     xf, yf = fourier(data[data.columns[each]].to_numpy(), N=N)
            #     ax.plot(xf, yf)
            # ax.set(
            #     xlim=(-1, 10)
            # )
            # plt.show()

            # ff = torch.fft.fft(torch.tensor(data[data.columns[1:7]].to_numpy()))
            # ff = (torch.pow(ff.real + ff.imag,.5))
            # print(ff)
            # plt.plot(ff)
            # plt.show()
            x = x_pd.to_numpy()
            # x_pd.legend(fontsize=40)
            # plt.figure(figsize=(16, 6))
            x_pd.plot(
                subplots=False, title=session, fontsize=16, figsize=(16, 6)
            )  # color= 'black'
            # plt.show()
            # print(data)

            T = 1
            fs = 50.0
            # if "150" in filename:
            #     lambda =
            # else:
            #     lambda_ = 1
            lambda_ = .35  # E2 => [.3 .4]
            if filename.split('_')[1] in ['E2', 'E4']:
                print("filename.split('_')[1]", filename.split('_')[1])
                lambda_ = .35
                WEIGHT = [1, .01, 1]
            elif filename.split('_')[1] in ['E3']:
                print("filename.split('_')[1]", filename.split('_')[1])
                lambda_ = 1
                WEIGHT = [1, .01, .8]
            else:
                lambda_ = .35

            counter = 0
            size = int(T * fs * (N / 1000.0) * lambda_ / 2)
            # change this: i-50 + i + i+50
            energy = cal_energy(size, x, weighted=WEIGHT)

            plt.plot(energy, 'r--', label="Energy")
            plt.legend(fontsize=16)
            # plt.plot(A, 'b--')
            # plt.legend(fontsize=)

            plt.show()

            confirmation = False
            begin = None
            end = None

            df = pd.DataFrame(data={
                'energy': energy
            })
            ilocs_min = argrelextrema(
                df.energy.values, np.less_equal, order=3
            )[0]
            ilocs_max = argrelextrema(
                df.energy.values, np.greater_equal, order=3
            )[0]
            print('min: ', ilocs_min)
            print('max: ', ilocs_max)

            while not confirmation:
                begin = int(input('begin: '))  # 77, 110,   184, 149
                end = int(input('end: '))  # 2722    3113   1968, 2427
                x_pd.plot(
                    subplots=False,
                    title=session,
                    fontsize=16,
                    figsize=(16, 6)
                )
                plt.plot(energy, 'r--', label="Energy")

                plt.axvline(int(begin), color='b', linestyle='-.')
                plt.axvline(int(end), color='b', linestyle='-.')
                # plt.figure(figsize=(8, 6))
                plt.legend(fontsize=16)
                plt.show()
                conf = input("confirmed? (1: confirm, 0: not confirm): ")
                if int(conf) == 1:
                    confirmation = True
            plt.figure(figsize=(16, 6))
            plt.plot(x[begin:end])
            print('starting points', begin, end)
            energy = cal_energy(size, x[begin:end], weighted=WEIGHT)

            plt.plot(energy, 'r--')
            plt.show()
            # plt.rcParams["figure.figsize"] = [7.50, 3.50]
            # plt.rcParams["figure.autolayout"] = True
            df = pd.DataFrame(data={
                'energy': energy
            })
            ilocs_min = argrelextrema(
                df.energy.values, np.less_equal, order=3
            )[0]
            ilocs_max = argrelextrema(
                df.energy.values, np.greater_equal, order=3
            )[0]
            print('min: ', ilocs_min)
            print('max: ', ilocs_max)
            for_x_pd = pd.DataFrame(x_pd[begin:end]).reset_index(drop=True)
            for_x_pd.plot(
                subplots=False,
                title=session,
                color='black',
                fontsize=16,
                legend=False
            )
            df.energy.plot(figsize=(16, 6), alpha=.3)

            # filter prices that are peaks and plot them differently to be visable on the plot

            df.iloc[ilocs_max].energy.plot(style='.', lw=10, color='red')
            df.iloc[ilocs_min].energy.plot(style='.', lw=10, color='green')
            print(type(df.iloc[ilocs_max].energy))
            for i, j in zip(
                df.iloc[ilocs_max].energy, df.iloc[ilocs_max].energy.index
            ):
                plt.text(j, i + 0.1, '{}'.format(j), color='red')

            for i, j in zip(
                df.iloc[ilocs_min].energy, df.iloc[ilocs_min].energy.index
            ):
                plt.text(j, i + 0.1, '{}'.format(j), color='green')

            # for i, v in enumerate(ilocs_max):
            #     plt.text(i, v + 25, "%d" % v, ha="center",  fontsize=12)

            # plt.legend()
            # plt.legend(fontsize=14)
            plt.show(block=False)
            x_copy = data.values[begin:end]  # x[begin:end]
            print(x_copy.shape)
            print('possible split:', ilocs_max[1:-1])
            if len(ilocs_max[1:-1]) == 9 and int(input("confirmed: ")) == 1:
                split = list(ilocs_max[1:-1])

            else:
                split = [int(input('split: ')) for i in range(NUM_OF_SPLIT)]

            split.append(len(x_copy))
            segments = []
            ind = 0
            print('split:', split)
            for i in range(NUM_OF_REPETITION):
                segments.append(x_copy[ind:split[i]])
                ind = split[i]

            fig, ax = plt.subplots(
                4, NUM_OF_REPETITION, sharey='row', figsize=(19.2, 9.77)
            )
            print(ax)
            xyz = ['null', 'x', 'y', 'z', 'x', 'y', 'z']
            for i in range(NUM_OF_REPETITION):
                print(segments[i][:, 1:7].shape)

                ax[0, i].plot(segments[i][:, 1:4])
                ax[1, i].plot(segments[i][:, 4:7])
                for each in range(1, 4):
                    N = len(segments[i][:, each])
                    xf, yf = fourier(segments[i][:, each], N=N)
                    ax[2, i].plot(xf, yf, label='acc_' + xyz[each])
                    ax[2, i].set_xlim(-1, 3)

                for each in range(4, 7):
                    N = len(segments[i][:, each])
                    xf, yf = fourier(segments[i][:, each], N=N)
                    ax[3, i].plot(xf, yf, label='gyro_' + xyz[each])
                    ax[3, i].set_xlim(-1, 3)

                ax[0, i].set_ylim(-3, 3)
                ax[1, i].set_ylim(-10, 10)
            plt.subplots_adjust(
                left=.125, bottom=.11, right=.9, top=.88, wspace=.2, hspace=.2
            )
            # plt.show()

            for i in range(NUM_OF_REPETITION):
                new_filename = filename.split('.')[0] + '_' + str(i) + '.csv'
                print('new filename:', new_filename)
                path = os.path.join(output_directory, new_filename)
                f = open(path, "w", newline='')
                writer = csv.writer(f)

                writer.writerow(HEADER)

                for row in segments[i]:
                    # print(row.shape)
                    writer.writerow(row)
                f.close()
            fig = plt.gcf()
            plt.show()
            # break
            # remove the file:
            subject = filename.split('_')[0][1:]
            exercise = filename.split('_')[1]
            rom = filename.split('_')[3]
            resist = filename.split('.')[0].split('_')[4]
            res = input("Confirm the result (1 yes or 0 no):")
            if int(res) != 1:
                continue
            else:
                looks_good = True
            # save the file and figure:
            fig.savefig(
                figure_directory + " ".join([subject, exercise, rom, resist]) +
                '.png',
                dpi=100
            )  # 1920x977
            with open(one_repetition_data_recording, 'a') as f:
                print(
                    "{}, [{}], {}, {}: [{}, {}]".
                    format(subject, exercise, rom, resist, begin, end) +
                    str(split),
                    file=f
                )

            os.remove(session)
            print('-------------', 'END', '-------------')
        reset_foldernames()
f.close()
# 1
#       120  150  30    60    90
# begin  77  110  184   149   132
# end   2722 3113 1968, 2427  2471

# 2
#       120  150  30    60    90
# begin  56  126  42   64   62
# end   1868 2433 1320, 1608  1716

# 3
#       120  150  30    60    90
# begin  27  33  36   24   43
# end   1372 1661 823, 1125  1160

# 134 272 403 531 655 784 910 1036 1155
# 114, 288, 453, 657, 811, 966, 1124, 1280, 1434
# 67, 148, 229, 305, 385, 463, 544, 625, 705
# 117, 227, 327, 434, 542, 649, 763, 871, 977,
# 95, 212, 335, 443, 560, 676, 793, 908, 1024,

# 4
#       120  150  30    60    90
# begin  59  53  137   48   100
# end   1534 2170 1400, 1348  1487
# 4 90: 103 212 317 438 548 673 791 918 1063

# 5
#       120  150  30    60    90
# begin  336  175  130   205   299
# end   1462 1692 856, 1096  1313

# 98, 208, 322, 430, 541, 664, 779, 902, 1033
# 134, 303, 447, 631, 770, 921, 1063, 1219, 1366,
# 67, 149, 220, 295, 367, 444, 519, 591, 660,
# 80, 172, 265, 364, 453, 549, 642, 725, 813,
# 88, 189, 311, 413, 516, 626, 724, 820, 928,

# 6
#       120  150  30    60    90
# begin  25  47  24   26   64
# end   1480 2219 921, 1168  1311

# 171, 317, 477, 616, 764, 901, 1044, 1169, 1304
# 240, 487, 732, 941, 1154, 1314, 1547, 1759, 1950,
# 90, 172, 258, 341, 428, 513, 606, 693, 778,
# 126, 237, 347, 457, 565, 677, 788, 894, 999,
# 91, 213, 340, 443, 551, 687, 817, 954, 1088,

# 7
#       120  150  30    60    90
# begin  42  64  45   31   57
# end   1102 1415 782, 819  968

# 112, 226, 339, 443, 554, 651, 754, 858, 961,
# 135, 305, 463, 599, 731, 861, 989, 1112, 1236,
# 76, 148, 225, 300, 372, 451, 523, 598, 665,
# 87, 174, 262, 340, 424, 506, 585, 668, 743,
# 84, 184, 279, 374, 467, 556, 650, 738, 825,

# 8
#       120  150  30    60    90
# begin  71  127  65   60   52
# end   1289 2580 785, 897  1157

# 116, 230, 344, 466, 593, 717, 837, 956, 1092,
# 328, 594, 848, 1089, 1316, 1551, 1773, 1997, 2221,
# 76, 150, 217, 284, 358, 426, 501, 576, 655,
# 84, 166, 245, 329, 411, 497, 577, 662, 744,
# 103, 214, 326, 435, 546, 659, 774, 879, 988,

# 9
#       120  150  30    60    90
# begin  101  126  77   82   61
# end   1484 1849 1150, 1276  1314

# 136, 271, 403, 552, 685, 834, 988, 1134, 1270,
# 186, 365, 560, 736, 903, 1058, 1243, 1414, 1580,
# 84, 190, 304, 412, 518, 632, 735, 852, 961,
# 156, 263, 374, 484, 596, 708, 828, 947, 1071,
# 142, 272, 397, 521, 625, 761, 880, 996, 1106,

# 10
#       120  150  30    60    90
# begin  0  0     25   82   61
# end   2123 2604 1437, 1276  1314

# [249, 459, 674, 886, 1092, 1297, 1505, 1710, 1905, 2123]
# [289, 592, 854, 1124, 1377, 1625, 1885, 2135, 2356, 2604]
# 84, 190, 304, 412, 518, 632, 735, 852, 961,
# 156, 263, 374, 484, 596, 708, 828, 947, 1071,
# 142, 272, 397, 521, 625, 761, 880, 996, 1106,

# 1 resistance band (1):

#       120  150  30    60    90
# begin   0   126   0    0    0
# end   1706 1849 1006, 1264  1300

# [198, 370, 538, 712, 872, 1034, 1194, 1344, 1518, 1706]
# N/A
# [102, 197, 302, 403, 505, 605, 700, 790, 880, 1006]
# [134, 232, 357, 477, 623, 761, 886, 1014, 1129, 1264]
# [198, 370, 538, 712, 872, 1034, 1194, 1344, 1518, 1706]
