import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.parameter import Parameter

from models.modules import module_encoder, MLP, Siamese, CNN_channel_encoder, channel_wise_attention
import torch.nn.functional as F
from torch import stft
class UnitNormClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weights'):
            w = module.weights.data
            w.div_(torch.norm(w, 6).expand_as(w))

class remove_padding(nn.Module):
    def __init__(self, opt):
        super(remove_padding, self).__init__()
    def forward(self, x):
        res = []
        for each_b in range(x.shape[0]):
            begin_ind = 1
            for i in range(x.shape[1]-1):
                if x[each_b, i, 0] != x[each_b, i+1, 0]:
                    break
                begin_ind += 1
            res.append(x[each_b, begin_ind::, :])
        return res

class _TrendGenerator(nn.Module):
    def __init__(self, expansion_coefficient_dim, target_length):
        super().__init__()

        # basis is of size (expansion_coefficient_dim, target_length)
        basis = torch.stack(
            [
                (torch.arange(target_length) / target_length) ** i
                for i in range(expansion_coefficient_dim)
            ],
            dim=1,
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class _SeasonalityGenerator(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        half_minus_one = int(target_length / 2 - 1)
        cos_vectors = [
            torch.cos(torch.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]
        sin_vectors = [
            torch.sin(torch.arange(target_length) / target_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]

        # basis is of size (2 * int(target_length / 2 - 1) + 1, target_length)
        basis = torch.stack(
            [torch.ones(target_length)] + cos_vectors + sin_vectors, dim=1
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)

class PhysiQ_classification(nn.Module):
    def __init__(self, opt):
        self.model_name = "physiq"
        opt.decoder_mode = "physiq"
        super().__init__()
        self.opt = opt
        self.explainable = opt.explainable_model
        self.channel_encoder = CNN_channel_encoder(opt)
        me = module_encoder(opt)
        self.weights = Parameter(torch.tensor([1.4, .8, 1.2, .9, 1.2, 1.3],
                                              requires_grad=True, device=opt.device))
        self.channel_attention = channel_wise_attention(opt)
        # if self.explainable:
        self.lstm = me.lstm
        self.cnn = me.cnn_encoder
        self.attention = me.attention
        self.device = me.device
        self.attention_flag = me.attention_flag
        self.sliding = me.sliding
        self.hidden_size = me.hidden_size
        # else:
        #     self.encoder = me
        self.mlp = MLP(opt, in_feature=opt.rnn_hidden_size)
        if opt.debug: print("In_feature", opt.cnn_hidden_size, opt.rnn_hidden_size, opt.hidden_size)

        self.trend_linear = nn.Linear(opt.hidden_size, 5)
        self.trend = _TrendGenerator(5, opt.total_length * 6)
        self.seasonality_linear = nn.Linear(opt.hidden_size, opt.total_length * 6-1)
        self.seasonality = _SeasonalityGenerator(opt.total_length * 6)
        self.residual = nn.Linear(opt.hidden_size, opt.total_length * 6)
    def forward_sliding_channel(self, inp):
        # i = self.sliding(inp)
        # zgm, zga = self.channel_attention(i)
        # importance = zgm + zga
        # r = torch.mean(importance, dim=1)
        # return r
        return self.weights.unsqueeze(dim=0).unsqueeze(dim=0)
    def forward(self, inp):

        i = self.sliding(inp)
        if self.opt.channel:
            # zgm, zga = self.channel_attention(i)
            # i_att = (zgm + zga) * i
            i_att = torch.mul(i, self.weights)
            x_t = i + i_att
        else:
            x_t = i
        # standard process for PhysiQ:
        segment_encoder_cnn = self.cnn(x_t)  # [8, 72]
        out = self.lstm(segment_encoder_cnn)
        encoder_output = out
        if self.attention_flag:
                encoder_output, attn_output_weights = self.attention(encoder_output, encoder_output, encoder_output)
        # print(encoder_output.shape)
        if not self.opt.explainable_model:
            out = self.mlp.forward(encoder_output)
            return out

        a = self.trend(self.trend_linear(encoder_output))
        b = self.seasonality(self.seasonality_linear(encoder_output))
        c = self.residual(encoder_output)
        rec = a + b + c
        # plt.plot(b.view(a.shape[0], 400, 6)[3, :, 0].cpu().detach().numpy())
        # plt.show()
        out = self.mlp.forward(encoder_output)
        return out, rec, a, b, c

class PhysiQ_siamese(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.model_name = "physiq_siamese"
        opt.decoder_mode = 'physiq_siamese'
        self.encoder = module_encoder(opt)
        self.siamese = Siamese(opt)
        # self.trend = _TrendGenerator(5, opt.total_length*6)
        # self.seasonality = _SeasonalityGenerator(opt.total_length*6)
        # self.residual = nn.Linear(opt.hidden_size, opt.total_length*6)

    def forward(self, inp0, inp1):
        encoder_output0 = self.encoder.forward(inp0)
        encoder_output1 = self.encoder.forward(inp1)
        out = self.siamese(encoder_output0, encoder_output1)
        return out.squeeze()

