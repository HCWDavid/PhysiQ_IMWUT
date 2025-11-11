import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        'Initialization'
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'

        assert len(self.y) == len(self.x), "length is not the same"
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        feature = self.x[index]
        target = self.y[index]
        if self.transform:
            # for i in range(feature.shape[1]):
            #     line = feature[:, i]
            #     fig = plt.subplot(6, 1, i + 1)
            #     fig.plot(line)
            # plt.show()
            feature = self.transform(feature)
            # for i in range(feature.shape[1]):
            #     line = feature[:, i]
            #     fig = plt.subplot(6, 1, i + 1)
            #     fig.plot(line)
            # plt.show()
        return feature, target


class SiameseNetworkDataset(torch.utils.data.Dataset):
    # make sure sampling deterministics: model see every combination of pair.
    def __init__(self, x, y, transform=None):
        'Initialization'
        self.x = x
        self.y = y
        self.x_length = len(x)
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.x[index][0]), self.transform(self.x[index][1]), torch.tensor([self.y[index]])
        return self.x[index][0], self.x[index][1], torch.tensor([self.y[index]])

    def __len__(self):
        return self.x_length
