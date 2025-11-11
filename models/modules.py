import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

from utils.util import calculate_num_sliding_windows


class sliding_windows(nn.Module):
    def __init__(self, opt):
        super(sliding_windows, self).__init__()
        self.total_length = opt.total_length
        self.width = opt.width
        self.step = opt.step

    def forward(self, input_time_series):
        return torch.swapaxes(input_time_series.unfold(-2, size=self.width, step=self.step), -2, -1)

    def get_num_sliding_windows(self):
        return round((self.total_length - (self.width - self.step)) / self.step)


class Siamese(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.linear = nn.Linear(calculate_num_sliding_windows(opt.total_length, opt.width, opt.step), 1)
        self.cosine_sim = torch.nn.CosineSimilarity(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input0, input1):
        features = self.cosine_sim(input0, input1)
        output = self.sigmoid(self.linear(features))
        return output


# class CNN_module(nn.Module):
#     def __init__(self, opt):
#         super(CNN_module, self).__init__()
#         self.dropout = opt.dropout
#         self.input_shape = opt.total_length
#         self.conv1 = nn.Conv1d(self.input_shape, self.input_shape // 6, 3)
#         self.pool = nn.MaxPool1d(2, 2)
#         self.conv2 = nn.Conv1d(self.input_shape // 6, opt.cnn_hidden_size, 2)
#         self.flatten = nn.Flatten()
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.dropout(x, p=self.dropout)
#         x = F.relu(self.conv2(x))
#         x = F.dropout(x, p=self.dropout)
#         x = self.flatten(x)
#         return x

class CNN_encoder_2D(nn.Module):
    def __init__(self, opt):
        super(CNN_encoder_2D, self).__init__()
        self.dropout = opt.dropout
        self.input_shape = opt.width
        self.conv1 = nn.Conv2d(self.input_shape, self.input_shape // 6, 3)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv2d(self.input_shape // 6, opt.cnn_hidden_size, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=self.dropout)

        x = self.flatten(x)
        return x


class CNN_encoder(nn.Module):
    def __init__(self, opt):
        super(CNN_encoder, self).__init__()
        self.dropout = opt.dropout
        self.input_shape = opt.width
        self.conv1 = nn.Conv1d(self.input_shape, self.input_shape // 6, 3)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(self.input_shape // 6, opt.cnn_hidden_size, 2)
        self.flatten = nn.Flatten()

    def forward(self, i):
        # print('i', i.shape)
        x = i.contiguous().view(-1, i.shape[2], i.shape[3])
        # print('reshape', x.shape)
        x = F.relu(self.conv1(x))
        # print('conv1', x.shape)

        x = self.pool(x)
        # print('after pool', x.shape)
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.conv2(x))
        # print('relu2', x.shape)
        x = F.dropout(x, p=self.dropout)
        # print('conv2', x.shape)
        x = self.flatten(x)
        x = x.view(i.shape[0], i.shape[1], -1)
        # print('res', x.shape)
        return x

class CNN_channel_encoder(nn.Module):
    def __init__(self, opt):
        super(CNN_channel_encoder, self).__init__()
        self.dropout = opt.dropout
        self.input_shape = opt.total_length
        self.conv1 = nn.Conv1d(self.input_shape, self.input_shape // 5, 3, padding=1)
        # self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(self.input_shape // 5, 10, 3, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(10, 1)

    def forward(self, i):
        # print('i', i.shape)
        x = i.view(-1, i.shape[1], i.shape[2])
        # print('reshape', x.shape)
        x = F.relu(self.conv1(x))
        # print('conv1', x.shape)

        # x = self.pool(x)
        # print('after pool', x.shape)
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.conv2(x))
        # print('relu2', x.shape)
        x = F.dropout(x, p=self.dropout)
        x = x.view(-1, x.shape[-1], x.shape[1])
        x = self.linear(x)
        x = x.squeeze()
        # print('conv2', x.shape)
        # x = self.flatten(x)
        # x = x.view(i.shape[0], i.shape[1], -1)
        # print('res', x.shape)
        return x
class lstm_wrapper(nn.Module):
    def __init__(self, opt):
        # TODO: RETURN out and h0_1 for all, so that we can interpret how each windows affect the results of it.
        super(lstm_wrapper, self).__init__()
        self.lstm = nn.LSTM(input_size=opt.cnn_hidden_size, hidden_size=opt.rnn_hidden_size,
                            num_layers=opt.num_layers, bidirectional=opt.bidirectional,
                            batch_first=opt.batch_first, dropout=opt.dropout)
        self.opt = opt

    def forward(self, inp):
        out, (h0_1, c0_1) = self.lstm(inp)  # [8, 1, 14]
        if self.opt.siamese:
            return out
        return h0_1[-1]
class module_encoder(nn.Module):
    # https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm#:~:text=The%20output%20state%20is%20the,step%20from%20the%20input%20sequence.
    # https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
    def __init__(self, opt):
        # hidden= 12 for cross validation # last used 72
        super().__init__()
        self.channel = opt.channel
        self.device = opt.device
        self.input_size = opt.input_size
        self.hidden_size = opt.rnn_hidden_size
        self.num_layers = opt.num_layers
        self.bidirectional = opt.bidirectional
        self.direction = 2 if opt.bidirectional else 1
        self.batch_size = opt.batch_size
        self.batch_first = opt.batch_first
        self.attention_flag = opt.attention_flag
        self.width = opt.width
        self.step = opt.step
        self.cnn_encoder = CNN_encoder(opt)
        # this lstm is used after cnn encoder:
        self.lstm = lstm_wrapper(opt)

        self.in_feature = self.hidden_size * (
            calculate_num_sliding_windows(opt.total_length, opt.width, opt.step))
        if opt.decoder_mode == 'siamese':
            self.in_feature = calculate_num_sliding_windows(opt.total_length, opt.width, opt.step)
        self.weights = Parameter(torch.tensor([1.4, .8, 1.2, .9, 1.2, 1.3],
                                              requires_grad=True, device=opt.device))
        self.sliding = sliding_windows(opt)
        if self.attention_flag:
            self.num_heads = opt.num_heads
            # assert self.lstm.direction * self.lstm.hidden_size % self.num_heads == 0, "Please make sure the number of head is divisible by the hidden feature of mha"
            self.attention = nn.MultiheadAttention(self.direction * self.hidden_size, opt.num_heads,
                                                   batch_first=self.batch_first)
        self.channel_attention = channel_wise_attention(opt)
    def sliding_windows_forward(self, input_x):
        x_t = self.sliding(input_x)
        return x_t
    def forward_sliding_channel(self, inp):
        i = self.sliding(inp)
        zgm, zga = self.channel_attention(i)
        importance = zgm + zga
        r = torch.mean(importance, dim=1)
        return r

    def _forward(self, x_t):
        segment_encoder_cnn = self.cnn_encoder(x_t)  # [8, 72]
        out = self.lstm(segment_encoder_cnn)
        segment = out
        if self.attention_flag:
            segment, attn_output_weights = self.attention(segment, segment, segment)
        return segment

    def forward(self, input_x):
        x_t = self.sliding(input_x)
        if self.channel:
            # zgm, zga = self.channel_attention(x_t)
            # i_att = (zgm + zga) * x_t
            # x_t = x_t + i_att
            i_att = torch.mul(x_t, self.weights)
            x_t = x_t + i_att

        segment_encoder_cnn = self.cnn_encoder(x_t)  # [8, 72]
        out = self.lstm(segment_encoder_cnn)
        segment = out
        if self.attention_flag:
            segment, attn_output_weights = self.attention(segment, segment, segment)
        return segment

    def init_hidden(self):
        rand_hidden = Variable(
            torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size))
        rand_cell = Variable(
            torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size))
        return rand_hidden, rand_cell


class MLP(nn.Module):
    def __init__(self, opt, in_feature):
        super().__init__()
        self.output_size = opt.output_size
        self.in_feature = in_feature
        self.input_fc = nn.Linear(self.in_feature, 50)  # 51
        self.hidden_fc = nn.Linear(50, 10)
        self.output_fc = nn.Linear(10, self.output_size)
        if opt.debug: print('self.output_size', self.output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        h_1 = F.leaky_relu(self.input_fc(x))
        h_2 = F.leaky_relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred

class channel_wise_attention(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.channel_wise_mlp = nn.Sequential(
            nn.Linear(6, 100),
            nn.Dropout(p=.5),
            nn.Linear(100, 6),
            nn.Sigmoid()
        )
        self.device = opt.device
    def forward(self, inp):
        global_avg = torch.empty(0).to(self.device)
        global_max = torch.empty(0).to(self.device)
        for index in range(inp.shape[1]):
            i_sub = inp[:, index, :, :]
            global_avg = torch.cat(
                [global_avg, torch.unsqueeze(F.avg_pool2d(i_sub, kernel_size=(i_sub.shape[1], 1)), dim=1)], dim=1)
            global_max = torch.cat(
                [global_max, torch.unsqueeze(F.max_pool2d(i_sub, kernel_size=(i_sub.shape[1], 1)), dim=1)], dim=1)
        # zgm, zga
        return self.channel_wise_mlp(global_max), self.channel_wise_mlp(global_avg)