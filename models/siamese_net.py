import random

import numpy as np
import torch
import torch.nn as nn
from scipy import spatial


class SiameseNetworkComparisonDataset(torch.utils.data.Dataset):

    def __init__(self, x1, x2):
        # x1: one session for the same exercise of x2: [segments, segments, segments, segments...]
        self.x1 = x1
        self.x2 = x2
        assert len(x1) == len(x2), "both of them should be the same length"

    def __getitem__(self, index):
        seg1 = self.x1[index]
        seg2 = self.x2[index]

        return seg1, seg2, 1.

    def __len__(self):
        return len(self.x1)


class SiameseNetworkDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, numberDataset=10000, initNormalize=False):
        'Initialization'
        self.numlabel = len(set(y))
        self.labelsDict = self._sortByLabels(x, y)
        self.total = numberDataset
        self.count = [0, 0]  # number of negative and positive
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # split data:

        self.sample = self._initSample(initNormalize)

    def __getitem__(self, index):
        return self.sample[index]

    def __len__(self):
        return self.total

    def _initSample(self, initNormalize):
        res = []
        if initNormalize:
            for i in range(self.total):
                res.append(self._getDistanceSample())
        else:
            for i in range(self.total):
                flip = random.choice([0, 1])
                if flip == 1:
                    res.append(self._getPositiveSample())
                    self.count[1] += 1
                else:
                    res.append(self._getNegativeSample())
                    self.count[0] += 1
        return res

    def _getDistanceSample(self):
        # getting the label:
        labelCategory = np.array([i for i in range(5)])
        labelDegree = [30, 50, 90, 120, 150]
        randomLabel1 = random.choice(labelCategory)
        randomLabel2 = random.choice(labelCategory)
        label = torch.tensor(labelCategory).float()
        s = labelCategory / len(labelCategory)
        seg1 = random.choice(self.labelsDict[randomLabel1])
        seg2 = random.choice(self.labelsDict[randomLabel2])
        # print(randomLabel1, randomLabel2, torch.tensor(labelDegree[randomLabel1]/180 - labelDegree[randomLabel2]/180))
        return seg1, seg2, torch.tensor(
            labelDegree[randomLabel1] / 180 - labelDegree[randomLabel2] / 180
        )  #s[randomLabel1] - s[randomLabel2]
        # getting the segments:

    def _getPositiveSample(self):
        # getting the label:
        labelCategory = [i for i in range(self.numlabel)]
        randomLabel = random.choice(labelCategory)

        # getting the segments:
        seg1 = random.choice(self.labelsDict[randomLabel])
        seg2 = random.choice(self.labelsDict[randomLabel])
        #         return seg1, seg2, torch.from_numpy(np.array([1],dtype=np.float32))
        return seg1, seg2, 1.

    def _getNegativeSample(self):
        # getting the label:
        labelCategory = [i for i in range(self.numlabel)]
        randomLabel1 = random.choice(labelCategory)
        labelCategory.remove(randomLabel1)
        randomLabel2 = random.choice(labelCategory)

        seg1 = random.choice(self.labelsDict[randomLabel1])
        seg2 = random.choice(self.labelsDict[randomLabel2])
        return seg1, seg2, 0.

    #         return seg1, seg2, torch.from_numpy(np.array([0],dtype=np.float32))

    def _sortByLabels(self, X, y):
        labelsDict = {
            k: []
            for k in range(0, self.numlabel)
        }

        print(labelsDict)
        for i in range(len(y)):
            index = y[i]
            labelsDict[index].append(X[i])
        # print('check', sum([len(k) for v, k in labelsDict.items()]))
        assert sum([len(k) for v, k in labelsDict.items()]) == len(X)
        return labelsDict


class LSTM(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=256,
        layer_dim=2,
        output_dim=7,
        dropout_prob=0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        #         out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


class SiameseNetwork(nn.Module):
    # , input_dim, hidden_dim, layer_dim, output_dim, dropout_prob
    def __init__(self, initNormalize=False):
        super(SiameseNetwork, self).__init__()
        self.lstm = LSTM()

        # depends on the features:
        # self.input_dim = 5 * self.lstm.hidden_dim
        self.input_dim = self.lstm.hidden_dim
        self.output_dim = 1 if initNormalize else 2
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.input_dim, int(self.input_dim / 2)),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(int(self.input_dim / 2), self.output_dim),
            #             nn.Linear(2, 1)
            nn.Tanh(),
        )

    #     def forward_once(self, x):
    #         # Forward pass

    #         h1, c1 = self.lstm.init_hidden(x)
    #         return h1, c1

    def forward(self, input1, input2):
        # without normalization of the time series:

        v1 = self.lstm(input1)
        v2 = self.lstm(input2)

        # features = torch.cat((v1, torch.abs(v1 - v2), v2, v1 * v2, (v1 + v2) / 2), 2)
        features = torch.abs(v1 - v2)

        output = self.classifier(features[:, -1, :])  # x.view(x.size()[0], -1)
        #         print('output', output.shape)
        #         out = output[:, -1, :]
        #         print(output.shape)
        # out = output
        #         m = nn.Sigmoid()
        #         out = m(output)
        # print(output)
        return output
