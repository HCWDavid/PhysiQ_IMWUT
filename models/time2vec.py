import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://github.com/ojus1/Time2Vec-PyTorch
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0

    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):

    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features,
                                                     1)).to(device)
        self.b0 = nn.parameter.Parameter(torch.randn(1)).to(device)
        self.w = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1)
        ).to(device)
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1)
                                        ).to(device)
        self.f = torch.sin

    def forward(self, tau):
        return t2v(
            tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0
        ).to(device)


class CosineActivation(nn.Module):

    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features,
                                                     1)).to(device)
        self.b0 = nn.parameter.Parameter(torch.randn(1)).to(device)
        self.w = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1)
        ).to(device)
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1)
                                        ).to(device)
        self.f = torch.cos

    def forward(self, tau):
        return t2v(
            tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0
        ).to(device)
