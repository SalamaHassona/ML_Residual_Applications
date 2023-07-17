import torch
from torch.utils import data
import numpy as np


class DynamicDataset(data.Dataset):

    def standardization(self, _X):
        _X_mean = _X.mean(dim=2).view(_X.shape[0], _X.shape[1], 1)
        _X_std = _X.std(dim=2).view(_X.shape[0], _X.shape[1], 1)
        X=(_X-_X_mean)/_X_std
        return X

    def __init__(self, xdata, labels, shuffle=False, batch_size=64):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.xdata = xdata
        self.labels = labels

    def __len__(self):
        return self.xdata.shape[0] // self.batch_size
        # return 5

    def __getitem__(self, index):
        self.batch_x = self.xdata[self.batch_size*index: self.batch_size*(index+1), :]
        self.batch_labels = self.labels[self.batch_size*index: self.batch_size*(index+1)]

        if self.shuffle:
            indexes = np.arange(0, self.batch_size)
            np.random.shuffle(indexes)
            self.batch_x = self.batch_x[indexes]
            self.batch_labels = self.batch_labels[indexes]

        X = self.standardization(torch.from_numpy(self.batch_x).type(torch.FloatTensor))
        labels = torch.from_numpy(self.batch_labels).type(torch.LongTensor)
        return X, labels