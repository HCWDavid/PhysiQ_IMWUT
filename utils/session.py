import math
import os
import os.path
# import torch
import warnings
from collections import Counter

# from time_visualization import drawGraph
# from transform_segments import transformTimeToSegments
# from model import *
import numpy as np
from matplotlib import pyplot
from pandas import DataFrame, read_csv
# from torch.fft import fft
from scipy import optimize
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

from utils.util import getStabilityBasedOnX

# stacks.iop.org/PM/39/075007/mmedia


def _isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def _most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


"""
TODO: Add a functionality of "Based on filename, get exercises label accordingly"
"""


class Sessions(object):

    def __init__(self, opt):
        self.sessions_file = []
        self._transform = opt.transform
        self.sessions = []
        self.subjects = set()
        self.sessions_file = []
        self.has_timestamp = opt.has_timestamp
        self.metrics = opt.metrics
        self.opt = opt
        if not opt.load:
            self.directory = opt.init_directory
            self._init_all_files()
            self.save(opt.input_filename)
        else:
            if opt.debug: print("ITS LOADING FROM BIN.........")
            self.directory = opt.load_directory
            self.filename = opt.input_filename
            self._load_all_files()

    def _init_all_files(self):
        for dirname, dirnames, filenames in os.walk(self.directory):
            for filename in tqdm(filenames):
                if filename in ['.DS_Store']:
                    continue
                session_dir_name = os.path.join(dirname, filename)
                session_dir = {
                    'dirname': dirname,
                    'filename': filename
                }
                self.sessions_file.append(session_dir_name)
                session = Session(
                    session_dir, has_timestamp=self.has_timestamp
                )
                self.sessions.append(session)
                self.subjects.add(session._subject)
        # if not self._load_sessions(threshold_percent, has_timestamp, DEBUG=DEBUG):
        #     print('POSSIBLE THRESHOLD CHECK: FAILED!')

    def _load_all_files(self):

        X, Y, subject, side, y_labels, side_labels, exercise, rom, resistance, id_repetition = self._load_data(
            self.directory, self.filename
        )
        self.sessions = []
        # load session data individually

        for x, y, subj, side, exer, rom, resist, id_rep in zip(
            X, Y, subject, side, exercise, rom, resistance, id_repetition
        ):
            self.subjects.add(subj)
            session = Session()
            session._x = x
            session._label = y
            session._subject = subj
            session._side = side
            session._exercise = exer
            session._ROM = rom
            session._resistance = resist
            session._ids = id_rep
            # session.load_data(x, y, subj, side, resist, id_rep, rom)
            self.sessions.append(session)
        return X, Y, subject, side, y_labels, side_labels, rom, id_repetition

    @staticmethod
    def _load_data(directory, filename):
        f = os.path.join(directory, filename)
        data = np.load(f, allow_pickle=True).item()

        x, y, subject = data['X'], data['y'], data['subject']
        # Handle both 'side' and 'hand' keys for compatibility
        side = data.get('side', data.get('hand', [None] * len(x)))
        side_labels = data.get('side_labels', data.get('hand_labels', None))

        def checkIfExist(name, dat):
            name = str(name)
            return dat[name] if name in data else [None] * len(x)

        y_labels = checkIfExist('exercise_labels', data)
        exercise = checkIfExist('exercise', data)
        ROM = checkIfExist('rom', data)
        # If ROM not in data but y looks like ROM labels (0-4), use y as ROM
        if ROM[0] is None and all(
            isinstance(val, (int, np.integer)) and 0 <= val <= 4
            for val in y
            if val is not None
        ):
            ROM = y
        resistance = checkIfExist('resistance', data)
        # Check for stability data
        if 'stability' in data:
            resistance = data['stability']
        id_repetition = checkIfExist('id_repetition', data)

        return x, y, subject, side, y_labels, side_labels, exercise, ROM, resistance, id_repetition

    def _get_data(self):
        x = []
        y = []
        subject = []
        exercise = []
        side = []
        rom = []
        resistances = []
        ids = []
        for session in self.sessions:
            x.append(session._x)
            y.append(session._label)
            subject.append(session._subject)
            exercise.append(session._exercise)
            side.append(session._side)
            rom.append(session._ROM)
            resistances.append(session._resistance)
            ids.append(session._ids)
        return x, y, subject, exercise, side, rom, resistances, ids

    def output_data(self):
        saving = dict()
        X, y, subject, exercise, side, rom, resistance, id_repetitions = self._get_data(
        )
        assert None in rom or max(rom) <= 4, "ROMs is wrong!"
        # labels: could be implemented later giving input:
        # X_labels = []
        side_labels = ['Left', 'Right']
        saving['X'] = X
        if self.metrics.lower() == 'rom':
            saving['y'] = rom
        elif self.metrics.lower() in ['resistance', 'stability', 'res', 'stb']:
            if self.opt: print("Y IS STABILITY")
            saving['y'] = resistance
        elif self.metrics.lower() in 'exercise':
            saving['y'] = exercise
        else:
            warnings.warn('Using default y as y label')
            saving['y'] = y
            # assert False, "Please identify the right metrics to load the data with y labeling accordingly"
        exercise_labels = ['', 'SA', 'ER', 'FF', 'IR']
        # rom_labels = ['30', '60', '90', '120', '150'] #TODO: this is for SA,
        # rom_labels = ['45', '90', '150']  # TODO: this is for ER
        saving['subject'] = subject
        saving['side'] = side
        # saving['X_labels'] = X_labels
        saving['exercise_labels'] = exercise_labels
        # saving['rom_labels'] = rom_labels
        saving['side_labels'] = side_labels
        saving['rom'] = rom
        saving['resistance'] = resistance
        saving['id_repetition'] = id_repetitions
        return saving

    def save(self, filename, directory='./bin'):
        f = os.path.join(directory, filename)
        saving = self.output_data()
        if self.opt.debug: print('saving...', filename)
        np.save(f, saving)
        return

    def count_all(self):
        for s in self.sessions:
            c = s.count()
            if c < 18 or c > 22:
                print(s.filename)

    def plot(self):
        for session in self.sessions:
            session.plot()
        return

    def find(self, subject=-1, resistance=None, rom=None, listOnly=False):
        assert resistance is None or type(
            resistance
        ) is list, "make sure the type of resistance you look for is list type"
        assert rom is None or type(
            rom
        ) is list, "make sure the type of rom you look for is list type"
        res = []
        for session in self.sessions:
            if (session.subject == subject or subject == -1) and (
                    resistance is None or session.resistance in resistance) \
                    and (rom is None or session._ROM in rom):
                res.append(session)
        if len(res) == 0:
            return None
        if listOnly:
            return res
        res_sessions = Sessions()
        res_sessions.sessions = res
        return res_sessions

    def find_by_subject(self, resistance=None, rom=None):
        """
        This find function requires:
         based on subject, the subject should have all the resistance and rom listed in parameter
        Parameters
        ----------

        resistance
        rom

        Returns
        -------

        """
        assert resistance is None or type(
            resistance
        ) is list, "make sure the type of resistance you look for is list type"
        assert rom is None or type(
            rom
        ) is list, "make sure the type of rom you look for is list type"
        res = []
        for subject in list(self.subjects):
            flag = any(
                [
                    self.find(subject=subject, resistance=[res], rom=[r])
                    is None for res in resistance for r in rom
                ]
            )
            if not flag:
                res.extend(
                    self.find(
                        subject=subject,
                        resistance=resistance,
                        rom=rom,
                        listOnly=True
                    )
                )
        res_sessions = Sessions()
        res_sessions.sessions = res
        return res_sessions

    def remove(self, subject):
        assert subject in self.subjects, "remove other subject"
        res_sessions = []

        for each in self.sessions:
            if each.subject != subject:
                res_sessions.append(each)
        self.sessions = res_sessions
        self.subjects.remove(subject)
        return self


class Session(object):

    def __init__(self, filename=None, has_timestamp=False):
        self._x = None
        # self.all_data = None # data in dataFrame
        self._label = None
        self._subject = None
        self._exercise = None
        self._side = None
        self._resistance = None
        self._ids = None
        self._ROM = None

        if filename is None:
            return
        filepath = os.path.join(filename['dirname'], filename['filename'])
        self.filename = filename['filename']

        self._subject, self._label, self._side, self._ROM, self._resistance, self._ids = self.process_filename(
            filename['filename']
        )

        with open(filepath, 'r') as f:
            _unprocessed = read_csv(f, delimiter=',', index_col=False)

        if has_timestamp:
            self.columns = _unprocessed.columns[1:7]
            self.timestamp = _unprocessed[_unprocessed.columns[0]]
        else:
            self.columns = _unprocessed.columns

        self.columns = self.columns[0:6]
        # processed the data into pandas
        self.all_data = _unprocessed[self.columns]

        self.np_data = np.array(self.all_data)
        self._x = np.array(self.all_data)
        # print(self.filename, self.np_data.shape)
        self._transformed = False
        self.amplified = self.amplify()

        ## using new ground truth to calculate resistance:
        self._resistance = getStabilityBasedOnX([self._x])[0]

    # def load_data(self, X, y, subject, side, resist, id_rep, ROM=None):
    #     self.np_data = X
    #     self.all_data = DataFrame(X)  # not sure if correct, need to testing it
    #
    #     # print(self.all_data.shape)
    #     # print(type(self.all_data))
    #     self.label = y
    #     self.subject = subject
    #     self.side = side
    #     self.resistance = resist
    #     self.id_repetition = id_rep
    #     self._ROM = ROM
    #     self._transformed = False
    #     self.amplified = self.amplify()
    #     return

    def process_filename(self, filename):
        filen = filename.split('.')[0].split('_')
        dict_side = {
            'L': 0.0,
            'R': 1.0
        }
        # print(filen)
        first = filen[0][1:]
        second = filen[1][1:]
        third = filen[2][0]
        # print(first, second, third)

        assert first.isdigit()
        subject = int(first)

        assert second.isdigit()
        exer = int(second)
        # L = 0, R = 1
        side = dict_side[third]
        rom = None
        id_repetition = None
        resistance = None
        if len(filen) > 3:
            fourth = filen[3][0:]
            if exer in [1, 3]:
                dict_rom = {
                    30: 0,
                    60: 1,
                    90: 2,
                    120: 3,
                    150: 4
                }
            elif exer in [2, 4]:
                dict_rom = {
                    45: 0,
                    90: 1,
                    120: 2,
                    150: 2
                }
            else:
                assert False, "This particular " + second + " dictionary labeling does not exist yet"
            assert fourth.isdigit()
            rom = dict_rom[int(fourth)]
        if len(filen) > 4:
            fifth = filen[4][0:]
            assert fifth.isdigit()
            resistance = int(fifth)
        if len(filen) > 5:
            lastth = filen[-1][
                0:]  # last one is the id_repetition if it is been segmented
            assert lastth.isdigit()
            id_repetition = int(lastth)

        return subject, exer, side, rom, resistance, id_repetition

    def plot(self, amp=False):

        self.all_data.plot(subplots=True, title=self.filename)

        pyplot.show()
        if amp:
            self.amplified.plot()
            pyplot.show()
        return

    def amplify(self):
        return self.all_data.sum(axis=1)

    @staticmethod
    def _target_func(x, a0, a1, a2, a3):
        return a0 * np.sin(a1 * x + a2) + a3

    def curve_fit(self, plot=True):
        # https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
        y = np.array(self.amplified)

        x = np.arange(len(y))

        fs = np.fft.fftfreq(len(x), x[1] - x[0])
        Y = abs(np.fft.fft(y))
        freq = abs(fs[np.argmax(Y[1:]) + 1])
        a0 = max(y) - min(y)  # AMPLITUDE
        a1 = 2 * np.pi * freq  # OMEGA
        a2 = 0  # PHASE
        a3 = np.mean(y)  # OFFSET
        p0 = [a0, a1, a2, a3]

        params, params_covariance = optimize.curve_fit(
            self._target_func, x, y, p0=p0
        )

        if plot:
            pyplot.scatter(x, y, label='Data', s=1)
            y_fit = [self._target_func(a, *params) for a in x]
            pyplot.plot(x, y_fit, label='Fitted function')
            pyplot.legend(loc='best')

            pyplot.show()
        return params, params_covariance

    def fourier(self):
        pass
        # s = fft(torch.from_numpy(amp_copy))

        # s = np.sin(amp_copy)
        # print(s.shape)
        # pyplot.plot(np.arange(len(s)), s)
        # # ax.plot(s)
        # pyplot.show()

    def count(self):

        y = np.array(self.amplified)
        x = np.arange(len(y))
        params, _ = self.curve_fit(plot=False)
        y_fit = self._target_func(x, *params)

        # maximum, _ = find_peaks(y_fit)
        # print(len(maximum))
        #
        # minimum, _ = find_peaks(y_fit*(-1.0))
        # print(len(minimum))

        aver = 0
        begin = y_fit[0]
        isIncreasing = y_fit[1] > begin
        prev = y_fit[1]
        for i in y_fit[2:]:
            if isIncreasing and prev <= begin <= i:
                aver += 1
            elif not isIncreasing and i <= begin <= prev:
                aver += 1
            prev = i
        countCompletedWave = aver

        print('Repetition is:', countCompletedWave)
        return countCompletedWave

    def stationarity(self, threshold_percent='5%', DEBUG=False):
        # Tests for trend non-stationarity
        # Null hypothesis is time series is non-stationary
        # 0: test statistics: more negative means more likely to be stationary
        # 1: test p value: if p-value is small (0.05), we reject the null hypothesis. reject non-stationary
        # 4: critical test statitics: different percentage for threshold of p-value
        checkpoints = [False] * len(self.np_data.T)
        assert threshold_percent in ['1%', '5%', '10%']
        for index in range(len(self.np_data.T)):
            each = self.np_data.T[index]
            test_statistics, p_value, _, _, criticals, _ = adfuller(each)
            if test_statistics < criticals[threshold_percent]:
                checkpoints[index] = True
        if DEBUG:
            if np.all(checkpoints):
                print('TEST PASSED!')
            else:
                print('TEST FAILED!!!!!')
        # adding this two line 281- 283
        # if not np.all(checkpoints):
        #     self.transform_diff()
        return np.all(checkpoints)

    def transform_diff(self):
        # X = yt - yt-1
        if self._transformed:
            return
        data = self.all_data
        data = data.diff()

        self.all_data = data.iloc[1:]

        self.np_data = np.array(self.all_data)

        self._transformed = True
        return

    def transform_log(self):
        if self._transformed:
            return

        data = self.all_data

        data = data.log()

        self.np_data = data

        self.np_data = np.array(self.all_data)

        self._transformed = True
        return

    # def transform_sqrt(self):
    #     if self._transformed:
    #         return
    #
    #     for each in self.columns:
    #         df = self.all_data[each]
    #         df = np.sqrt(df)
    #         self.all_data[each] = df
    #
    #     self.np_data = np.array(self.all_data)
    #
    #     self._transformed = True
    #     return
    #
    # def transform_shift(self):
    #     if self._transformed:
    #         return
    #
    #     for each in self.columns:
    #         df = self.all_data[each]
    #         df = df.shift(1) / df
    #         self.all_data[each] = df
    #     self.all_data = self.all_data.iloc[1:]
    #     self.np_data = np.array(self.all_data)
    #
    #     self._transformed = True
    #     return
