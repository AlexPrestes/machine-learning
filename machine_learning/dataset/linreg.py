import numpy as np
import torch
from torch.utils.data import Dataset


class Reta(Dataset):
    def __init__(self, a, b, size, transform=None, target_transform=None):
        self.a = a
        self.b = b
        self.size = size

        self._generation_data()

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X.reshape(-1, 1), y.reshape(-1, 1)
    
    def _generation_data(self):
        X = np.linspace(0, 1, self.size)
        y = self.a*X +self.b
        y += (0.5 -np.random.rand())*y/2

        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)