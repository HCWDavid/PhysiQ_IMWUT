import math
import os
import random
import shutil
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from time import sleep

import numpy as np
import seaborn as sns
import torch
from joblib import dump, load
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

MODE = ['rom', 'repetition', 'rep_1toN', "rom_repetition", "stability", 'stb']
MAX_REPETITION = 5

# sport: > 3 times a week = 2
#        < 0 times a week = 1
#        else = 0

# injury: > 3 times a week = 2
#        < 0 times a week = 1
#        else = 0
# gender: 0 for male

def metrics(label1, label2):
    return 1 - abs(label1 - label2)


def getDistance(ROM_1, ROM_2):
    """
    Regression Metrics to calculate two labels
    Parameters
    ----------
    ROM_1: the first range of motion labels [0,1,2,3,4]
    ROM_2: the second range of motion labels [0,1,2,3,4]

    Returns
    -------
    float; the calculation of distance of two labels
    """
    labelDegree = [30, 50, 90, 120, 150]  # [0,1,2,3,4]
    return metrics(labelDegree[ROM_1] / 180, labelDegree[ROM_2] / 180)
    # return 1 - abs(labelDegree[ROM_1] / 180 - labelDegree[ROM_2] / 180)


def getRepDiff(REP_1, REP_2):
    MaxRepetition = MAX_REPETITION  # max of the repetition of given exercises
    return metrics(REP_1 / MaxRepetition, REP_2 / MaxRepetition)


def getDistanceStability(label_1, label_2):
    labelDegree = [0, 1, 2]  # TODO: binary stability
    # print("TESTING THE LABEL RESULTS:", label_1, label_2)
    return metrics(labelDegree[label_1] / 2, labelDegree[label_2] / 2)


def get_stability(x):
    fs = 50  # sample rate, Hz
    cutoff = 20  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2

    def butter_lowpass_filter(data, cutoff, fs, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)

    res = 0.0
    for d in x.T:
        y_ = butter_lowpass_filter(d, cutoff, fs, order)
        res += signaltonoise(y_)
    return abs(res / 6)


def getStabilityBasedOnX(X):
    res_y = []
    for each in X:
        r = get_stability(each)
        r = math.tanh(r)
        res_y.append(r)
    return res_y


def getDistanceStabilityBasedOnX(x_1, x_2):
    return metrics(get_stability(x_1), get_stability(x_2))


def getDistAndRepDiff(Y_1, Y_2):
    # (repetition, rom), (repetition, rom)
    return (getRepDiff(Y_1[0], Y_2[0]) + getDistance(Y_1[1], Y_2[1])) / 2


def normalization(x, length=-1):
    y_normed = deepcopy(x)

    for s in y_normed:
        for index in range(len(s.T)):
            data = s.T[index]
            s.T[index] = (data - np.mean(data)) / np.std(data)
    if length <= 0:
        maxlength = max([len(x.T[0]) for x in y_normed])
    else:
        maxlength = length
    x_interped = np.zeros((len(y_normed), maxlength, 6))
    for jndex in range(len(y_normed)):
        x = y_normed[jndex]
        for index in range(len(x.T)):
            data = x.T[index]
            #         print(data)
            x_interped[jndex, :, index] = np.interp(
                np.linspace(0, 1, maxlength), np.linspace(0, 1, len(data)),
                data
            )
    # print(x_interped.shape)
    return x_interped


def train_valid_test_split(X, y, subject, test_size=0.2):
    """
    simply split the data by X_train, X_valid, X_test, y_train, y_valid, y_test
    Parameters
    ----------
    X
    y
    subject
    test_size

    Returns
    -------

    """
    # X_train, X_valid, X_test, y_train, y_valid, y_test = [],[],[],[],[],[]
    # X_train, X_valid, X_test, y_train, y_valid, y_test = X[0:leng], X[leng::leng+50], X[leng+50::], y[0:leng], y[leng::leng+50],  y[leng+50::]
    temp = defaultdict(list)
    for i in range(len(subject)):
        sub = subject[i]
        temp[sub].append(i)
    temp_list = list(temp.keys())
    test_sample_size = int(len(temp_list) * test_size)
    assert len(temp_list) - 1 > test_sample_size >= 1, \
        "you need to have a correct number of test_size to get subject testing, validating and training data"
    valid_sample_size = 1

    test_sample = random.sample(temp_list, test_sample_size)
    for element in test_sample:
        temp_list.remove(element)
    valid_sample = random.sample(temp_list, valid_sample_size)
    for element in valid_sample:
        temp_list.remove(element)
    train_sample = temp_list

    def output(temp_sample):
        print(temp_sample)
        resx = []
        resy = []
        for index in temp_sample:
            resx += [X[j] for j in temp[index]]
            resy += [y[j] for j in temp[index]]
        print(len(resx), len(resy))
        return resx, resy

    X_train, y_train = output(train_sample)
    X_valid, y_valid = output(valid_sample)
    X_test, y_test = output(test_sample)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_by_subject(
    X, y, subject, subjectTosplit, valid_size=.1, test_size=.1, shuffle=False
):
    after_test_split_valid_size = valid_size / (1.0 - test_size)
    x_subject = []
    y_subject = []
    s_subject = []
    for i in range(len(subject)):
        if subject[i] == subjectTosplit:
            x_subject.append(X[i])
            y_subject.append(y[i])
            s_subject.append(subjectTosplit)
    x_subject = np.array(x_subject)
    y_subject = np.array(y_subject)
    s_subject = np.array(s_subject)
    X_train, X_test, y_train, y_test, tr_s, te_s = train_test_split(
        x_subject, y_subject, s_subject, test_size=test_size, shuffle=shuffle
    )
    X_train, X_valid, y_train, y_valid, tr_s, va_s = train_test_split(
        X_train,
        y_train,
        tr_s,
        test_size=after_test_split_valid_size,
        shuffle=shuffle
    )

    # print('subject', subjectTosplit)
    # print(X_train.shape)
    # print(X_valid.shape)
    # print(X_test.shape)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, tr_s, va_s, te_s


def dump_scaler(ss, directory='./scalers/', mode='default'):
    # assert len(ss) == 1, "only one scaler works for now"
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = [i for i in os.listdir(directory) if os.path.isfile(os.path.join(directory, i)) and \
             mode in i]
    f = get_scaler_name(mode, str(len(files)), directory=directory)
    dump(ss, f)


def get_scaler_name(mode, index, directory='./scalers/'):
    f = os.path.join(directory, mode + '_ss_' + str(index) + '.scaler')
    return f


def load_scaler(file):
    # assert len(ss) == 1, "only one scaler works for now"
    assert os.path.exists(file), "file does not exists"
    return load(file)


def pair_up_by_subject(X, y, subject, shuffle=False, mode='rom'):
    """
    pair up by its subject: combination pair of inputs
    Parameters
    ----------
    X
    y
    subject
    shuffle
    mode

    Returns
    -------

    """
    # print('in pair_up_subject', X.shape)
    # print('in pair_up_subject subject', len(subject))
    x_group_by_subject = defaultdict(list)
    y_group_by_subject = defaultdict(list)

    for index in range(len(subject)):
        key = subject[index]
        x_group_by_subject[key].append(X[index])
        y_group_by_subject[key].append(y[index])
    # print(y_group_by_subject)
    res_x, res_y, res_s = [], [], []
    for key in list(set(subject)):
        key_x = x_group_by_subject[key]
        key_y = y_group_by_subject[key]
        # shuffle each person's list of exercises before put combination the data
        if shuffle:
            temp = list(zip(key_x, key_y))
            random.shuffle(temp)
            key_x, key_y = zip(*temp)
        z, w = pair_up(key_x, key_y, mode=mode)

        res_x.extend(z)
        res_y.extend(w)
        res_s.extend([key] * len(w))
    # print('beefore ending in pair_up_subject', len(res_x), 'length of pair up', len(x_group_by_subject))
    return res_x, res_y, res_s


def pair_up_on_new_data(
    X,
    y,
    subject=None,
    ss=None,
    shuffle=False,
    pad_method='front',
    printOn=False,
    mode='rom',
    longest=None,
    value='none'
):
    assert ss is not None, "Please provide the scaler"
    if longest is None:
        longest = get_longest_length(X)
    else:
        pass
    # pad the rest to nan
    X = pad_all_to_longest(
        X, value=value, longest=longest, pad_method=pad_method
    )
    X_test = X
    y_test = y

    s_test = subject

    X_test = transform(np.array(X_test), ss, printOn=printOn)
    # print("scaler info:", ss[0].mean_, ss[0].scale_, ss[0].var_)
    X_test, y_test, _ = pair_up_by_subject(
        X_test, y_test, s_test, shuffle=shuffle, mode=mode
    )

    X_test = np.nan_to_num(X_test)

    # print("DEBUG:")
    # print("after pair up:")
    # print("X_test", X_test.shape)
    return np.array(X_test), np.array(y_test)


def pair_up_train_valid_test_split(
    X,
    y,
    subject=None,
    valid_size=.2,
    test_size=.1,
    shuffle=False,
    pad_method='front',
    value='none',
    debug=False,
    mode='rom',
    longest=None,
    pair_up=False,
    include_subjects=False
):
    """
    Standardize train-valid-test split based on the size of valid and test
    for each subject get valid_size and test_size split out

    normalize on training data, transform valid and test

    pair them up by training data, validing data, testing data FOR EACH SUBJECT (combination on each subject)

    Parameters
    ----------
    X
    y
    subject
    valid_size
    test_size
    shuffle
    front
    debug
    mode
    longest

    Returns
    -------

    """
    if longest is None:
        longest = get_longest_length(X)

    else:
        pass
    # pad the rest to nan
    X = pad_all_to_longest(
        X, value=value, longest=longest, pad_method=pad_method
    )
    # print('before 0.0', X.shape)
    # X = fit_transform(X, ss)

    X_train = []
    X_valid = []
    X_test = []

    y_train = []
    y_valid = []
    y_test = []

    s_train = []
    s_valid = []
    s_test = []

    for i in set(subject):
        current_x_train, current_x_valid, current_x_test, \
        current_y_train, current_y_valid, current_y_test, \
        current_tr_s, current_va_s, current_te_s = \
            split_by_subject(X, y, subject, i, valid_size=valid_size, test_size=test_size, shuffle=shuffle)
        X_train.extend(list(current_x_train))
        X_valid.extend(list(current_x_valid))
        X_test.extend(list(current_x_test))

        y_train.extend(list(current_y_train))
        y_valid.extend(list(current_y_valid))
        y_test.extend(list(current_y_test))

        s_train.extend(list(current_tr_s))
        s_valid.extend(list(current_va_s))
        s_test.extend(list(current_te_s))

    ss = generate_scalers(1)

    X_train = fit_transform(np.array(X_train), ss, printOn=debug)

    X_valid = transform(np.array(X_valid), ss, printOn=debug)

    X_test = transform(np.array(X_test), ss, printOn=debug)
    # print("scaler info:", ss[0].mean_, ss[0].scale_, ss[0].var_)
    dump_scaler(ss)
    if pair_up:
        X_train, y_train, _ = pair_up_by_subject(
            X_train, y_train, s_train, shuffle=shuffle, mode=mode
        )
        X_valid, y_valid, _ = pair_up_by_subject(
            X_valid, y_valid, s_valid, shuffle=shuffle, mode=mode
        )
        X_test, y_test, _ = pair_up_by_subject(
            X_test, y_test, s_test, shuffle=shuffle, mode=mode
        )

    X_train = np.nan_to_num(X_train)
    X_valid = np.nan_to_num(X_valid)
    X_test = np.nan_to_num(X_test)
    if include_subjects:
        assert pair_up is not True, "can not return subjects info with pair_up turned on!"
        return np.array(X_train), np.array(y_train), np.array(
            X_valid
        ), np.array(y_valid), np.array(X_test), np.array(y_test), np.array(
            s_train
        ), np.array(s_valid), np.array(s_test)
    return np.array(X_train), np.array(y_train), np.array(X_valid), np.array(
        y_valid
    ), np.array(X_test), np.array(y_test)


def train_test_split_rom(
    X,
    y,
    subject=None,
    test_size=0.2,
    numOfRepetition=10,
    shuffle=False,
    each_split=False
):
    assert 0 <= test_size < 1.0

    if each_split:
        assert subject is not None
        X_temp = X
        y_temp = y
        if shuffle:
            temp = list(zip(X, y, subject))
            random.shuffle(temp)
            X_temp, y_temp, subject = zip(*temp)

        res = defaultdict(list)
        for i, s in enumerate(zip(subject, y_temp), 0):
            # print(s)
            res[s].append(i)
        res = dict(res)
        num_pick = int(numOfRepetition * test_size)

        X_test = []
        y_test = []
        X_train = []
        y_train = []

        for k, v in res.items():
            random.shuffle(v)
            for each in v[0:num_pick]:
                # print(v[0: num_pick], each)
                X_test.append(X_temp[each])
                y_test.append(y_temp[each])
            for each in v[num_pick:len(v)]:
                X_train.append(X_temp[each])
                y_train.append(y_temp[each])
            # X_test.append(X_temp[0: num_pick])
            # y_test.append(y_temp[0: num_pick])
            # X_test.append(X_temp[num_pick:len(v)])
            # y_test.append(y_temp[0: num_pick])

        ## unittest:
        te = defaultdict(int)
        for i in y_test:
            te[i] += 1
        for i in y_train:
            te[i] += 1
        for value in dict(te).values():
            assert value == 90

        if shuffle:
            temp = list(zip(X_train, y_train))
            random.shuffle(temp)
            X_train, y_train = zip(*temp)

        return X_train, X_test, y_train, y_test

    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)


def seed(sd=0, cudnn=False, deterministic=True):
    np.random.seed(sd)
    random.seed(sd)
    torch.manual_seed(sd)
    torch.backends.cudnn.benchmark = cudnn
    torch.cuda.manual_seed(sd)
    torch.backends.cudnn.deterministic = deterministic


def calc_out_layer(in_channel, padding=0, dilation=1, kernel_size=3, stride=1):
    return (
        in_channel + 2 * padding - dilation * (kernel_size - 1) - 1
    ) / stride - 1


def augmentation(X, y, lengthToAugment):
    assert len(lengthToAugment) > 0


def pair_up(x, y, mode='rom'):
    """
    This function has an assumption of labelDegree = [30, 50, 90, 120, 150]
    it yet does not work with activity recognition.
    It pairs up every single combination of data:
        nCr = len(x)_Combination_2
    Parameters
    ----------
    y: labels of Range of Motion
    x: data input as array (list) to combine the data into pair

    Returns
    -------
    [(seg1, seg2)], [labels]
    """
    assert mode in MODE, "Please choose correct mode"
    segment_index_pairs = []
    segment_pairs = []
    labels = []
    if mode == 'rep_1toN':
        # 1-2, 1-3, 1-4, 1-5... (for repetition comparison)
        store_one_rep_index = []
        store_else_index = []
        for index in range(len(y)):
            element = y[index]
            if element == 1:
                # these are repetition of 1:
                store_one_rep_index.append(index)
            store_else_index.append(index)
        for i in store_one_rep_index:
            for j in store_else_index:
                segment_index_pairs.append((i, j))
                segment_pairs.append((x[i], x[j]))
                labels.append(getRepDiff(y[i], y[j]))

    else:
        for i, j in combinations(list(range(len(x))), 2):
            segment_index_pairs.append((i, j))
            segment_pairs.append((x[i], x[j]))
            if mode == 'rom':
                labels.append(getDistance(y[i], y[j]))
            elif mode in ["repetition", 'rep']:
                labels.append(getRepDiff(y[i], y[j]))
            elif mode == 'rom_repetition':
                labels.append(getRepDiff(y[i], y[j]))
            elif mode in ["stability", 'stb']:
                labels.append(metrics(y[i], y[j]))
            else:
                assert False, "The mode/metric is wrong"
                # labels.append(getDistanceStabilityBasedOnX(x[i], x[j]))
                # labels.append(getDistanceStability(y[i], y[j]))
        # print(len(segment_pairs), len(labels))
    return segment_pairs, labels


def pad_all_to_longest(X, value='none', longest=None, pad_method='front'):
    """

    Parameters
    ----------
    longest: explicit pad all X to this length, but has to be longer or equal to the X's longest
    value: represents the value to be padded with, default NaN
    pad_method: padding to the front, back, or middle
    X: the input X data
    Returns
    -------
    find the longest, and zero padding for every dataset to the front or the back
    """
    # print("has longest padding in front or back")
    # if longest is None or longest == -1:
    #     long = np.max([len(x) for x in X])
    # else:
    assert pad_method in ['front', 'back', 'middle'], "pad method is incorrect"
    long = longest
    if np.max([len(x) for x in X]) > longest:
        long = np.max([len(x) for x in X])
    shortest = np.min([len(x) for x in X])
    # print('shortest', shortest)
    for index in range(len(X)):
        each = np.array(X[index])
        fillIn = np.zeros((long - each.shape[0], each.shape[1]))
        if value == 'none':
            fillIn[:] = np.nan
        elif value == 'random':
            fillIn[:] = np.random.rand(long - each.shape[0], each.shape[1])
        elif type(value) is str:
            fillIn[:] = eval(value)
            assert type(eval(value)) is int, "Input is wrong"
        else:
            assert False, "wrong"
        if pad_method == 'front':
            X[index] = np.vstack([fillIn, each])
        elif pad_method == 'back':
            X[index] = np.vstack([each, fillIn])
        else:  # 'middle'
            # fillIn0 = np.zeros(((long - each.shape[0])//2, each.shape[1]))
            # fillIn1 = np.zeros((math.ceil((long - each.shape[0])/2), each.shape[1]))
            fillIn0, fillIn1 = np.array_split(fillIn, 2)
            X[index] = np.vstack([fillIn0, each, fillIn1])
        # print(X[index].shape)
    return np.array(X)


def get_longest_length(X):
    return np.max([len(x) for x in X])


def normalization_on_axis(X):
    # https://www.binarystudy.com/2021/09/how-to-normalize-pytorch-tensor-to-0-mean-and-1-variance.html
    # https://discuss.pytorch.org/t/understanding-transform-normalize/21730
    import torch.nn.functional as F
    from torchvision import transforms
    temp_X = deepcopy(X)
    temp = torch.tensor([[], [], [], [], [], []]).T
    for each in temp_X:
        each = torch.tensor(each)
        temp = torch.cat([temp, each])
    denom = temp.norm(2, 0, keepdim=True).clamp_min(1e-12)
    # print(denom.shape)
    for i in range(len(X)):
        each = torch.tensor(X[i])
        m_img = torch.mean(each, -2, keepdim=True)
        std_img = torch.std(each, -2, keepdim=True)
        X[i] = (torch.tensor(X[i]) - m_img) / std_img
        # print(torch.std_mean(X[i], unbiased=False))
    # for i in range(len(temp.T)):
    #     print(torch.std_mean(temp[:, i], unbiased=False))
    return X


def generate_scalers(num=6):
    # TODO: explicit mention what scaler to use for evaluation
    """

    Parameters
    ----------
    num
    the number of scalers = the number of axis, sum of (accelerometer and gyroscope)
    Returns
    -------
    [StandardScaler]
    """
    return [StandardScaler() for _ in range(num)]


def fit_transform(X, ss: [StandardScaler], num=6, printOn=False):
    """
    fit the training dataset
    Parameters
    ----------
    X
    ss
    num
    printOn

    Returns
    -------

    """
    # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
    import warnings

    ## NEW: try use only one ss:
    s = ss[0]

    temp = deepcopy(X)
    reshaped_temp = np.concatenate(temp, axis=0)
    reshaped_back = np.array(np.array_split(reshaped_temp, len(temp), axis=0))
    assert np.all(
        np.isclose(reshaped_back, X, equal_nan=True)
    ), "reshape back they should be equal"

    reshaped_X = np.concatenate(X, axis=0)
    reshaped_X = s.fit_transform(reshaped_X)

    # for i in range(num):
    #     print('fit transform: mean {:6f}, std {:6f}'.format(np.nanmean(reshaped_X[:, i]), np.nanstd(reshaped_X[:, i])))
    reshaped_X = np.array_split(reshaped_X, X.shape[0], axis=0)
    reshaped_X = np.array(reshaped_X)
    # reshaped_X = reshaped_X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    # reshaped_X = new_reshaped_X
    if printOn:
        for i in range(num):
            print(
                'fit transform: mean {:6f}, std {:6f}'.format(
                    np.nanmean(reshaped_X[:, :, i]),
                    np.nanstd(reshaped_X[:, :, i])
                )
            )
    return reshaped_X
    # print(np.reshape(X, (400*331, 6)).shape)

    # for i in range(num):
    #     # warnings.filterwarnings("ignore", category=RuntimeWarning)
    #     print(X[:, :, i].shape)
    #     X[:, :, i] = ss[i].fit_transform(X[:, :, i])
    #     print('fit transform: mean {:6f}, std {:6f}'.format(np.nanmean(X[:, :, i]), np.nanstd(X[:, :, i])))
    #     warnings.filterwarnings("default", category=RuntimeWarning)
    # return X


def transform(X, ss: [StandardScaler], num=6, printOn=False):
    """
    this is used for testing or validation dataset, it is to transform the dataset based on fitted training dataset
    Parameters
    ----------
    X
    ss
    num
    printOn

    Returns
    -------

    """
    s = ss[0]
    reshaped_X = np.concatenate(X, axis=0)
    reshaped_X = s.transform(reshaped_X)

    # for i in range(num):
    #     print('fit transform: mean {:6f}, std {:6f}'.format(np.nanmean(reshaped_X[:, i]), np.nanstd(reshaped_X[:, i])))

    reshaped_X = np.array_split(reshaped_X, X.shape[0], axis=0)
    reshaped_X = np.array(reshaped_X)
    # reshaped_X = reshaped_X.reshape(X.shape[0], X.shape[1], X.shape[2])
    # reshaped_X = new_reshaped_X
    if printOn:
        for i in range(num):
            print(
                'fit         : mean {:6f}, std {:6f}'.format(
                    np.nanmean(X[:, :, i]), np.nanstd(X[:, :, i])
                )
            )
    return reshaped_X


def cross_validation(
    X,
    Y,
    subject,
    subjectToTest=None,
    pad_method='front',
    test_size=0.05,
    mode='rom',
    longest=None,
    pair_up=True,
    value='none'
):
    """
    Normalization before padding, then split the dataset based on cross-validation on given subject
    Parameters
    ----------
    mode
    test_size
    X: dataset of all X
    Y: label
    subject: its corresponding subject
    subjectToTest: which subject to test, currently only support integer, meaning 1 subject at a time
    front: padding to the front or the back

    Returns
    -------
    train_x, train_y, valid_x, valid_y, test_x, test_y
    """
    print("MODE", mode)

    if subjectToTest is not None and subjectToTest in set(subject):
        print('ssssssssssssssssssssss', len(X))
        train_x = []
        train_y = []
        valid_x = []
        valid_y = []
        test_x = []
        test_y = []
        index_group_by_subject = defaultdict(list)
        x_group_by_subject = defaultdict(list)
        y_group_by_subject = defaultdict(list)
        # if longest is None or longest < 0:
        # longest = get_longest_length(X)
        print("LONGEST: USING INPUT", get_longest_length(X))
        ss = generate_scalers(1)

        # store info by subject:
        for i in range(len(subject)):
            sub = subject[i]
            # print(sub)
            # if sub == 9:
            #     print(len(x_group_by_subject[sub]))
            #     print(X[i])
            index_group_by_subject[sub].append(i)
            x_group_by_subject[sub].append(X[i])
            y_group_by_subject[sub].append(Y[i])

        # normalize the train+valid dataset
        # TODO: previous code:
        lengthOfDataPerSubject = len(
            x_group_by_subject[sorted(list(set(subject)))[0]]
        )  # originaly it is x_group_by_subject[1] whyyyyyyy
        # print("lengthOfDataPerSubject", lengthOfDataPerSubject)
        forWhichXes_train = []
        forWhichSubjects_train = []
        non_testing_idx = -1
        for k, v in x_group_by_subject.items():
            # print(k)
            if k != subjectToTest:
                forWhichXes_train.extend(v)
                forWhichSubjects_train.append(k)
                non_testing_idx = k
        # lengthOfDataPerSubject = len(
        #     x_group_by_subject[sorted(list(set(subject)))[non_testing_idx]])
        print("lengthOfDataPerSubject", lengthOfDataPerSubject)
        # print("len of forWhichXes", len(forWhichXes_train))
        # print("longest length", longest)
        # pad the rest to nan
        forWhichXes_train = pad_all_to_longest(
            forWhichXes_train,
            value=value,
            longest=longest,
            pad_method=pad_method
        )
        # print('before 0.0', forWhichXes_train.shape)
        forWhichXes_train = fit_transform(forWhichXes_train, ss)
        # pad back nan to 0.0
        forWhichXes_train = np.nan_to_num(forWhichXes_train)
        # print('after 0.0', forWhichXes_train.shape)
        print("saving the scaler...")
        dump_scaler(ss, mode=mode)

        index = 0
        for i in range(len(forWhichSubjects_train)):
            k = forWhichSubjects_train[i]
            # print(k)
            x_group_by_subject[k] = forWhichXes_train[index:index +
                                                      lengthOfDataPerSubject]
            index += lengthOfDataPerSubject

        # normalize the test dataset and pad to NaN

        print("transformed subject", str(subjectToTest), end=' ')
        temp_test_x = x_group_by_subject[subjectToTest]
        temp_test_x = pad_all_to_longest(
            temp_test_x, value=value, longest=longest, pad_method=pad_method
        )
        temp_test_x = transform(temp_test_x, ss)

        # Nan to 0.0
        temp_test_x = np.nan_to_num(temp_test_x)
        x_group_by_subject[subjectToTest] = list(temp_test_x)

        # before pair up:
        test_sub = []
        train_sub = []
        # print("SUBJECT", set(subject))
        # print(x_group_by_subject.keys())
        for key in sorted(list(set(subject))):
            # print("KEY", key)
            key_x = x_group_by_subject[key]
            key_y = y_group_by_subject[key]
            # shuffle each person's list of exercises before put combination the data
            temp = list(zip(key_x, key_y))
            random.shuffle(temp)
            key_x, key_y = zip(*temp)
            if key == subjectToTest:
                if pair_up:
                    test_x, test_y = pair_up(key_x, key_y, mode=mode)
                else:
                    test_x, test_y = key_x, key_y
                test_sub = [key] * len(test_x)
            else:
                if pair_up:
                    z, w = pair_up(key_x, key_y, mode=mode)
                else:
                    z, w = list(key_x), list(key_y)
                # print(len(key_y))
                train_x.extend(z)
                train_y.extend(w)
                train_sub.extend([key] * len(z))

        train_x, valid_x, train_y, valid_y, train_sub, valid_sub = train_test_split(
            train_x, train_y, train_sub, test_size=test_size, shuffle=True
        )
        print('length', len(train_x), len(valid_x), len(test_x))
        print('WHAT')
        return np.array(train_x), np.array(train_y), np.array(
            valid_x
        ), np.array(valid_y), np.array(test_x), np.array(test_y), np.array(
            train_sub
        ), np.array(valid_sub), np.array(test_sub)


def calculate_num_sliding_windows(length, width, step):
    mod = (length - width) % step
    res = (length - width) / step
    if mod > 0:
        return int(math.ceil((length - width) / step))
    elif mod == 0:
        return int(math.ceil((length - width) / step)) + 1
    else:
        print(length, width, step)
        print("IN ELSEEEEE")
        return int(res - 1)


def get_num_sliding_windows(opt):
    return round((opt.total_length - (opt.width - opt.step)) / opt.step)


# 123 345 567 789
# 123 345 567 789 91011  (11-3)/


def subject_validation(saving, subjectToTest):
    X = saving['X']
    Y = saving['y']
    subject = saving['subject']
    subjectX = []
    subjectY = []
    subjectS = []

    for x, y, s in zip(X, Y, subject):
        if subjectToTest == s:
            subjectX.append(x)
            subjectY.append(y)
            subjectS.append(s)

    trainx, tempx, trainy, tempy, trains, temps = train_test_split(
        subjectX, subjectY, subjectS, test_size=.2
    )

    validx, testx, validy, testy, valids, tests = train_test_split(
        tempx, tempy, temps, test_size=.5
    )

    return trainx, trainy, validx, validy, testx, testy, trains, valids, tests


def standardize(train_data):
    train_data = torch.tensor(np.array(train_data))
    means = train_data.mean(dim=1, keepdim=True)
    stds = train_data.std(dim=1, keepdim=True)
    normalized_data = (train_data - means) / stds
    return normalized_data


def get_dictionary_by_subject_by_y(
    x, y, subject, id_repetition=None, indexOnly=False
):
    dic = dict()  # {subject: {ROM: []} }
    for i in set(subject):
        dic[i] = dict()
        for j in set(y):
            dic[i][j] = []
    if id_repetition is None:
        for i, (each) in enumerate(zip(x, y, subject)):
            key_subj = each[2]
            key_rom = each[1]
            if indexOnly:
                dic[key_subj][key_rom].append(i)
            else:
                dic[key_subj][key_rom].append(each[0])
    else:
        print("id is not None")
        assert indexOnly is not True, "in pre-defined id_repetition, please do not set indexOnly to true"
        # pre-define all spaces in the list with number of id_rep
        for i in set(subject):
            dic[i] = dict()
            for j in set(y):
                dic[i][j] = [None] * len(set(id_repetition))

        for each in zip(x, y, subject, id_repetition):
            key_subj = each[2]
            key_rom = each[1]
            index = each[3]
            dic[key_subj][key_rom][index] = each[0]

    return dic


def get_num_exercise(
    dic, SUBJECT=None, ROM=None, num=3, sameSample=False, adjacent=True
):
    """
    it will return concat the exercise depends on the num: [][][] if num=3
    Parameters
    ----------
    adjacent
    dic: the dictionary contains all information about the input x, use get_dictionary_by_subject_by_y() to generate the dict
    SUBJECT
    ROM
    num
    sameSample: if True, all sample will be the same for "num" of time

    Returns
    -------

    """
    get_sample = random.choice(list(dic.keys()))  # (index, x_value)
    get_rom = random.choice(list(dic[get_sample].keys()))

    if ROM is not None:
        get_rom = ROM
    if SUBJECT is not None:
        get_sample = SUBJECT

    pair = random.choice(list(enumerate(dic[get_sample][get_rom])))
    x_t = pair[1]
    id_t = pair[0]
    res = np.array(x_t)
    ids = [SUBJECT, ROM, id_t]
    if sameSample:
        res = np.tile(res, (num, 1))
        ids = [id_t] * num

    else:
        if adjacent:
            maxlen = len(dic[get_sample][get_rom]) - 1
            contains = [id_t]
            for _ in range(num - 1):
                choices = []
                if 0 < np.min(contains):
                    choices.append(np.min(contains) - 1)
                if np.max(contains) < maxlen:
                    choices.append(np.max(contains) + 1)
                id_new_t = random.choice(choices)
                x_t = dic[get_sample][get_rom][id_new_t]
                if id_new_t < id_t:
                    res = np.vstack([x_t, res])
                    contains = [id_new_t] + contains
                else:
                    res = np.vstack([res, x_t])
                    contains = contains + [id_new_t]
                id_t = id_new_t
            ids = [SUBJECT, ROM] + contains

        else:
            for _ in range(num - 1):
                get_rom = random.choice(list(dic[get_sample].keys()))
                pair = random.choice(list(enumerate(dic[get_sample][get_rom])))
                x_t = pair[1]
                id_t = pair[0]
                res = np.vstack([res, x_t])
                ids.append(id_t)
    # print('result shape:', res.shape)
    ids = "-".join([str(i) for i in ids])
    return res, ids


def _get_multi_samples(
    x,
    y,
    subject,
    SUBJECT=None,
    numSamples=100,
    num_repetition=3,
    ROM=None,
    id_repetition=None,
    adjacent=True,
    onlyRep=True
):
    """
    a detailed explaintation of implementation has been explained in the get_multi_samples function()
    Parameters
    ----------
    x
    y
    subject
    numSamples
    num_repetition
    ROM
    onlyRep: return only the num_repetition size, any lengths less than this will not return.
    Returns
    -------

    """
    temps = []  # result of x
    temps_labels = []  # result of y of repetition
    temps_roms = []  # result of y of Range of Motion
    temps_subject = []  # result of subject
    multi_ids = [
    ]  # this ids is used to check if there are same id of repetition i.g. same merged exercises repeated of the same number

    subject_category = set(subject)
    rom_category = set(y)
    id_category = set(id_repetition)
    dic = get_dictionary_by_subject_by_y(
        x, y, subject, id_repetition=id_repetition
    )

    TEMP_RESULT = defaultdict(list)
    if SUBJECT is not None:
        subject_category = [SUBJECT]
    for each_subject in subject_category:
        # number of merging, [], [][], [][][],...up to num_repetition, each [] is one segment of repetition of exercise.
        begin_repetition = num_repetition if onlyRep else 1

        for rep in (range(begin_repetition, num_repetition + 1)):
            max_numSamples = len(id_category) - (rep - 1) if adjacent else len(
                id_category
            )**rep  # updated: Not anymore => ASSUMPTION! this assumption is made to assume that for each subject, each range of motion, there is 10 segments/repetitions
            # print(max_numSamples)
            if ROM is None:
                # for len(rom_category) range of motions each rom has max num of samples:
                max_numSamples *= len(rom_category)
            if max_numSamples > numSamples:
                max_numSamples = numSamples
            # print(max_numSamples)
            counter = 0
            # repeat random selections of num of sample or number of max sample can have:
            while counter < max_numSamples:
                # print("?")
                if ROM is not None:
                    assert ROM in rom_category, "ROM is not in the right category, please make sure they are in the range of " + str(
                        rom_category
                    )
                    res, ids = get_num_exercise(
                        dic=dic,
                        num=rep,
                        SUBJECT=each_subject,
                        ROM=ROM,
                        adjacent=adjacent
                    )
                    if ids not in multi_ids:
                        temps.append(res)
                        multi_ids.append(ids)
                        TEMP_RESULT[each_subject].append(res)
                        temps_subject.append(each_subject)
                        temps_labels.append(rep)
                        counter += 1
                else:
                    # if ROM None, pick FROM THE SAME ROM FOR EACH REPETITION TO CONCAT (NO "FROM DIFFERENT ROMs" are GOING TO PUT TOGETHER)
                    for rom in rom_category:
                        res, ids = get_num_exercise(
                            dic=dic,
                            ROM=rom,
                            num=rep,
                            SUBJECT=each_subject,
                            adjacent=adjacent
                        )
                        if ids not in multi_ids:
                            # print("ids is", ids)
                            temps.append(res)
                            multi_ids.append(ids)
                            TEMP_RESULT[each_subject].append(res)
                            temps_subject.append(each_subject)
                            temps_labels.append(rep)
                            temps_roms.append(rom)
                            counter += 1

    return temps, temps_labels, temps_roms, temps_subject


def get_raw_samples(
    x,
    y,
    subject,
    SUBJECT=None,
    ROM=None,
    numSamples=100,
    num_repetition=3,
    id_repetition=None,
    mode='repetition',
    adjacent=True,
    onlyRep=True
):
    assert mode in MODE, "Please choose correct mode"

    temps, temps_labels, temps_roms, temps_subject = _get_multi_samples(
        x,
        y,
        subject,
        SUBJECT=SUBJECT,
        numSamples=numSamples,
        num_repetition=num_repetition,
        ROM=ROM,
        id_repetition=id_repetition,
        adjacent=adjacent,
        onlyRep=onlyRep
    )

    if mode in ['repetition', 'rep_1toN']:
        y = temps_labels
    elif mode in ['rom']:
        y = temps_roms
    elif mode in ['rom_repetition']:
        y = temps_labels  # list(zip(temps_labels, temps_roms))
    else:
        raise AssertionError("please choose the right mode")
    return temps, y, temps_subject  # x, y, subject


def get_multi_samples(
    x,
    y,
    subject,
    SUBJECT=None,
    ROM=None,
    numSamples=100,
    num_repetition=3,
    sameSample=False,
    shuffle=True,
    front=True,
    test_size=.2,
    id_repetition=None,
    adjacent=True,
    mode='repetition',
    onlyRep=False,
    longest=None
):
    """

    Samples are been processed depends on the mode, and a pair-up process will be down after the raw samples get collected.
    This is being done in the randomly collecting fashion.

    mode will depends which y (label) will be used for the model: [rom, repetition(rep_1toN)]

    for each subject:
        for each [1 to num_repetition]:
            for numSamples:
                concat sample with 1 to num_repetition, i.e. num_repetition=3: [1], [2].... [1][2].... [1][2][3], each [] is one repetition exercise


    Parameters
    ----------
    id_repetition
    x
    y
    subject
    ROM
    numSamples
    num_repetition
    sameSample
    shuffle
    front
    test_size
    mode

    Returns
    -------

    """
    temps, y, temps_subject = get_raw_samples(
        x,
        y,
        subject,
        SUBJECT=SUBJECT,
        ROM=ROM,
        numSamples=numSamples,
        num_repetition=num_repetition,
        id_repetition=id_repetition,
        mode=mode,
        adjacent=adjacent,
        onlyRep=onlyRep
    )
    print(set(y))

    # print("DEBUG:")
    # print("before pair up:")
    # print("total number of samples", len(temps))
    # print("each subject has", len(temps) / len(set(temps_subject)))

    x_train, y_train, x_valid, y_valid, x_test, y_test = pair_up_train_valid_test_split(
        temps,
        y,
        temps_subject,
        valid_size=.1,
        test_size=test_size,
        front=front,
        mode=mode,
        shuffle=shuffle,
        longest=longest
    )
    # print(x_valid.shape, y_valid.shape)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def find(x, y, subject, particularSubject):
    """
    based on x and y and subject, find the particular subject and return its x and y
    Parameters
    ----------
    particularSubject
    x
    y
    subject

    Returns
    -------

    """
    res_x = []
    res_y = []
    res_s = []
    everything_else_x = []
    everything_else_y = []
    everything_else_s = []
    for a, b, s in zip(x, y, subject):
        if s == particularSubject:
            res_x.append(a)
            res_y.append(b)
            res_s.append(s)
        else:
            everything_else_x.append(a)
            everything_else_y.append(b)
            everything_else_s.append(s)

    return res_x, res_y, res_s, everything_else_x, everything_else_y, everything_else_s


def plot_confusion_matrix(target, predict, dir=''):
    plt.rcParams.update({
        'font.size': 22
    })
    plt.figure(figsize=(10, 10), dpi=300)
    cm = confusion_matrix(target, predict)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cmn, annot=True, cmap='Blues')
    sns.set(font_scale=8.0)

    # ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('Predicted Values', fontdict={
        'size': '18'
    })
    ax.set_ylabel('Actual Values ', fontdict={
        'size': '18'
    })

    ## Ticket labels - List must be in alphabetical order
    # ax.xaxis.set_ticklabels(['False', 'True'])
    # ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.

    # plt.show()
    plt.savefig(os.path.join(dir, 'graph0.png'), dpi=80)
    return


# after_test_split_valid_size = valid_size / (1.0 - test_size)
#
# index_group_by_subject = defaultdict(list)
# x_group_by_subject = defaultdict(list)
# y_group_by_subject = defaultdict(list)
# longest = get_longest_length(X)
# ss = generate_scalers(6)
#
# print("longest length", longest)
# # pad the rest to nan
# # forWhichXes_train = X
# X = pad_all_to_longest(X, longest=longest, front=front)
# print('before 0.0', X.shape)
# X = fit_transform(X, ss)
#
# # # pad back nan to 0.0
# X = np.nan_to_num(X)
# print('after 0.0', X.shape)
# # X = forWhichXes_train
# # store info by subject:
# for i in range(len(subject)):
#     sub = subject[i]
#     index_group_by_subject[sub].append(i)
#     x_group_by_subject[sub].append(X[i])
#     y_group_by_subject[sub].append(y[i])
# print("num of subject in groupby", len(x_group_by_subject), len(y_group_by_subject))
# new_x = []
# new_y = []
# for key in list(set(subject)):
#     print('subject', key)
#     key_x = x_group_by_subject[key]
#     key_y = y_group_by_subject[key]
#     # shuffle each person's list of exercises before put combination the data
#     if shuffle:
#         temp = list(zip(key_x, key_y))
#         random.shuffle(temp)
#         key_x, key_y = zip(*temp)
#     z, w = pair_up(key_x, key_y)
#     new_x.extend(z)
#     new_y.extend(w)
# print("total", len(new_x), len(new_y))
# X_train, X_test, y_train, y_test = train_test_split(new_x, new_y, test_size=test_size, shuffle=shuffle)
# print()
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=after_test_split_valid_size,
#                                                       shuffle=shuffle)
# print(len(X_train), len(X_valid), len(X_test))
# # print(y_test)


def get_scaler_directory(model_name):
    directory = './scalers/'
    # isExist = os.path.exists(directory)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    print("scaler dir:", directory)
    return directory


def check_directory_and_create(dir):
    CHECK_FOLDER = os.path.isdir(dir)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(dir)
    return


def get_pth_directory(model_name):
    txt_directory = 'pth/' + model_name + '_pth'
    isExist = os.path.exists(txt_directory)
    if not isExist:
        os.makedirs(txt_directory)
    return txt_directory


def get_scripts_directory(model_name):
    """
    create the scripts directory
    Parameters
    ----------
    model_name

    Returns
    -------

    """
    txt_directory = 'scripts/' + model_name + '_scripts'
    isExist = os.path.exists(txt_directory)
    if not isExist:
        os.makedirs(txt_directory)
    return txt_directory


def get_classification_str(classification=False):
    return "./results" + (
        "_classification" if classification else "_regression"
    ) + "/"


def create_specific_directory(
    model_name, name="", filename="", classification=False
):
    import datetime
    import os
    result = get_classification_str(classification)
    if filename == "":
        mydir = result + name + "_" + model_name + "_" + datetime.datetime.now(
        ).strftime('%Y_%m_%d_%H-%M-%S')
    else:
        mydir = result + filename
    os.makedirs(mydir)
    return mydir


def get_specific_directory(
    model_name, name="", filename="", classification=False
):
    import os
    result = get_classification_str(classification)
    if filename == "":
        all_subdirs = [
            result + d
            for d in os.listdir(result)
            if os.path.isdir(result + d) and 'rom' in d and 'standard' in d
        ]
        mydir = max(all_subdirs, key=os.path.getmtime)
    else:
        mydir = result + filename
    return mydir


def save_paths(mydir, model_name, name='', classification=False):
    result = get_classification_str(classification)
    isExist = os.path.exists(result)
    if not isExist:
        os.makedirs(result)

    # scripts txt directory:
    # txt_directory = os.path.join(mydir, get_scripts_directory(model_name))
    txtName = "_".join([model_name, name]) + '.txt'
    txtName = os.path.join(mydir, txtName)
    # mydir = "/results/" + model_name + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')

    # saved model pth directory:
    # pth_directory = os.path.join(mydir, get_pth_directory(model_name))
    pthName = "_".join([model_name, name]) + '.pth'
    pthName = os.path.join(mydir, pthName)
    return txtName, pthName


def interpolate_one_sample(x, length):
    x_interped = np.zeros((length, 6))
    for index in range(len(x.T)):
        data = x.T[index]
        x_interped[:, index] = np.interp(
            np.linspace(0, 1, length), np.linspace(0, 1, len(data)), data
        )
    return x_interped


def interpolate(X, Y, subject, length, byAddLength=0, excludeSubject=-1):
    y_normed = deepcopy(X)
    lengthlist = [len(x.T[0]) for x in y_normed]
    maxlength = max([len(x.T[0]) for x in y_normed])
    if length == -1:
        length = maxlength
    assert length >= maxlength, "please make sure the length is greater than max length"
    res = []
    Y_res = []
    s_res = []
    for jndex in range(len(y_normed)):
        if subject[jndex] == excludeSubject:
            continue
        x = y_normed[jndex]
        if byAddLength != 0:
            length = len(x) + byAddLength
        Y_res.append(Y[jndex])
        res.append(interpolate_one_sample(x, length))
        s_res.append(subject[jndex])
    return res, Y_res, s_res


def delete(X, Y, subject, byDeleteLength=0, excludeSubject=-1):
    y_normed = deepcopy(X)
    lengthlist = [len(x.T[0]) for x in y_normed]
    maxlength = max([len(x.T[0]) for x in y_normed])
    avg = int(np.average(lengthlist))
    res = []
    Y_res = []
    s_res = []
    for jndex in range(len(y_normed)):
        if subject[jndex] == excludeSubject:
            continue
        x = y_normed[jndex]
        if byDeleteLength != 0:
            sub = int(10 * (len(x) - byDeleteLength) / len(x))
        else:
            sub = 9
        #         if len(x) < avg:
        #             res.append(x)
        #         else:
        res.append(x[np.mod(np.arange(x.shape[0]), 10) < sub, ...])
        Y_res.append(Y[jndex])
        s_res.append(subject[jndex])

    return res, Y_res, s_res


def downsample(X, Y, subject, R=2, excludeSubject=-1):
    res = []
    Y_res = []
    s_res = []
    for jndex in range(len(X)):
        if subject[jndex] == excludeSubject:
            continue
        x = X[jndex]
        sam_res = []
        for index in range(len(x.T)):
            b = x.T[index]
            pad_size = math.ceil(float(b.size) / R) * R - b.size
            b_padded = np.append(b, np.zeros(pad_size) * np.nan)
            c = np.nanmean(b_padded.reshape(-1, R), axis=1)
            sam_res.append(c)
        res.append(np.array(sam_res).T)
        Y_res.append(Y[jndex])
        s_res.append(subject[jndex])
    return res, Y_res, s_res


# print(x_interped.shape)


def data_interp(X, Y, subject, excludeSubject=-1):
    print('data interp', excludeSubject)
    # addedlength = random.randint(25, 100+1)
    # addedlength2 = random.randint(25, 100 + 1)
    # a = interpolate(X, Y, subject, length=-1)
    res_x, res_y, res_s = [], [], []
    for addedlength in [50]:
        aprime = interpolate(
            X,
            Y,
            subject,
            length=-1,
            byAddLength=addedlength,
            excludeSubject=excludeSubject
        )
        x = aprime[0]
        y = aprime[1]
        s = aprime[2]
        res_x.extend(list(x))
        res_y.extend(list(y))
        res_s.extend(list(s))

    for downsamp in [2]:
        aprime = downsample(
            X, Y, subject, R=downsamp, excludeSubject=excludeSubject
        )
        x = aprime[0]
        y = aprime[1]
        s = aprime[2]
        res_x.extend(list(x))
        res_y.extend(list(y))
        res_s.extend(list(s))

    # # aprimep = interpolate(X, Y, subject, length=-1, byAddLength=addedlength2, excludeSubject=excludeSubject)
    # b = delete(X, Y, subject, byDeleteLength=50)

    res_x.extend(X)
    res_y.extend(Y)
    res_s.extend(subject)
    print('the size is :', len(res_x))
    return res_x, list(res_y), list(res_s)


def data_flip_augment(res, ref, mode='ROM'):
    if mode.lower() in ["stability", 'resistance']:
        mode = 'resistance'
    elif mode.lower() == "rom":
        mode = 'ROM'
    else:
        assert False
    res_x, res_y, res_s, res_id = [], [], [], []
    res_x.extend(res['X'])
    res_y.extend(res[mode])
    res_s.extend(res['subject'])
    res_id.extend(res['id_repetition'])
    print(set(res['subject']))
    intersec = set(res['subject']).intersection(set(ref['subject']))

    def sort_ses(ses):

        lis = list(
            zip(ses['X'], ses[mode], ses['subject'], ses['id_repetition'])
        )
        s = sorted(lis, key=lambda x: (x[1], x[2], x[3]))
        res = []
        for i in range(len(lis)):
            if lis[i][2] in intersec:
                res.append(lis[i])
        return res

    lis_ref = sort_ses(ref)
    lis = sort_ses(res)

    for index, (l, r) in enumerate(zip(lis_ref, lis)):
        # acc X, Z:
        for aixs_i in [0, 2, 4]:
            r = l[0][:, aixs_i]
            lis_ref[index][0][:, aixs_i] = np.average(r) - r

    X, y, s, id_r = zip(*lis_ref)
    res_x.extend(list(X))
    res_y.extend(list(y))
    res_s.extend(list(s))
    res_id.extend(list(id_r))
    return res_x, res_y, res_s, res_id


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    # https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb
    xx = (
        np.ones((X.shape[1], 1)) *
        (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))
    ).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    #     cs_x = CubicSpline(xx[:,0], yy[:,0])
    #     cs_y = CubicSpline(xx[:,1], yy[:,1])
    #     cs_z = CubicSpline(xx[:,2], yy[:,2])
    res = [CubicSpline(xx[:, i], yy[:, i])(x_range) for i in range(len(X.T))]
    return np.array(res).transpose()


def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)


def data_mag_warp(X, Y, subject, id_repetition, excludeSubject=-1, sigma=.2):
    res = []
    Y_res = []
    s_res = []
    id_res = []
    for jndex in range(len(X)):
        if subject[jndex] == excludeSubject:
            continue
        x = X[jndex]
        da_x = DA_MagWarp(x, sigma)
        res.append(da_x)
        Y_res.append(Y[jndex])
        s_res.append(subject[jndex])
        id_res.append(id_repetition[jndex])
    return res, Y_res, s_res, id_res


def usage_augmentations(
    X, y, subject, id_repetition, opt, excludeSubject=-1, sigma=.2, R=2
):
    res_x, res_y, res_s, res_id = [], [], [], []
    res_x.extend(list(X))
    res_y.extend(list(y))
    res_s.extend(list(subject))
    res_id.extend(list(id_repetition))
    if opt.mag_warp:
        res = data_mag_warp(
            X, y, subject, id_repetition, excludeSubject, sigma
        )
        res_x.extend(res[0])
        res_y.extend(res[1])
        res_s.extend(res[2])
        res_id.extend(res[3])

    return res_x, res_y, res_s, res_id


class mag_warping(object):

    def __call__(self, sample):
        # print(sample)
        return DA_MagWarp(sample, sigma=.2)
