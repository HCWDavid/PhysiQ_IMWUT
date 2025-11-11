import torch
import torch.nn as nn
import torchvision.models as models


class sliding_windows(nn.Module):

    def __init__(self, opt):
        super(sliding_windows, self).__init__()
        self.width = opt.width
        self.step = opt.step

    def forward(self, input_time_series):
        return torch.swapaxes(
            input_time_series.unfold(-2, size=self.width, step=self.step), -2,
            -1
        )


class LSTM_baseline(nn.Module):

    def __init__(self, opt):
        super(LSTM_baseline, self).__init__()
        self.model_name = "LSTM_baseline"
        self.backbone = nn.LSTM(
            input_size=opt.input_size,
            hidden_size=opt.hidden_size,
            num_layers=opt.num_layers,
            bidirectional=opt.bidirectional,
            batch_first=opt.batch_first,
            dropout=opt.dropout
        )
        self.linear = nn.Linear(opt.width * opt.hidden_size, 5)
        self.sliding = sliding_windows(opt)
        self.hidden_size = opt.hidden_size

    def _run(self, input_x):
        x_t = self.sliding(input_x)

        # segment = None
        segment_att = None

        for i in range(x_t.shape[1]):
            each = x_t[:, i, :, :]
            out, h_0 = self.backbone(each)  # [8, 1, 14]

        return out

    def forward(self, x1):
        x1 = self._run(x1)
        x1 = x1.reshape(x1.shape[0], -1)
        return nn.Sigmoid()(self.linear(x1))


class LogisticRegression(torch.nn.Module):

    def __init__(self, opt):
        super(LogisticRegression, self).__init__()
        self.model_name = "logistic_regression"
        self.linear = torch.nn.Linear(
            opt.input_size * opt.total_length, opt.output_size
        )

    def forward(self, x):
        #         print(x.shape)
        x = x.view(x.shape[0], -1)
        outputs = torch.sigmoid(self.linear(x))
        #         print(outputs.shape)
        return outputs


import torch.nn.functional as F


class CNN_baseline(nn.Module):

    def __init__(self, opt, output_size=2):
        super().__init__()
        self.input_shape = opt.total_length  # 1 is the cnn hidden layer

        # layers:
        self.conv1 = nn.Conv1d(self.input_shape, 16, opt.cnn_kernel_size)
        self.batch1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(2)
        self.num_fc = calc_out_layer(
            self.input_shape, kernel_size=opt.cnn_kernel_size
        )
        self.fc = nn.Linear(32, opt.output_size)
        self.model_name = "CNN"

    def forward(self, inp):
        inp = self.conv1(inp)
        inp = self.batch1(inp)
        inp = F.tanh(inp)
        inp = self.max_pool1(inp)
        inp = F.dropout(inp, .2)
        inp = nn.Flatten()(inp)
        return self.fc(inp)


def calc_out_layer(in_channel, padding=0, dilation=1, kernel_size=3, stride=1):
    return int(
        (in_channel + 2 * padding - dilation *
         (kernel_size - 1) - 1) / stride - 1
    )
