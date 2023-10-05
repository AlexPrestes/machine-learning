import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

class LinearRegress(nn.Module):
    def __init__(self, input_layer=1):
        super().__init__()
        self.layer = nn.Linear(input_layer, 1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        y = self.layer(x)
        return y

    def _split_train_val(self, dataset_training, train_rate = 0.7):
        n = len(dataset_training)
        train_size = int(train_rate*n)
        val_size = int((1-train_rate)*n)
        return random_split(dataset_training, [train_size, val_size])
    
    def _training(self, dataset_training, num_batches):
        dl_training = DataLoader(dataset_training, num_batches, shuffle=True)
        for batch, (X, y) in enumerate(dl_training):
            # Compute prediction and loss
            pred = self(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()/num_batches

    def _validation(self, dataset_validation, num_batches):
        dl_val = DataLoader(dataset_validation, num_batches, shuffle=True)
        with torch.no_grad():
            loss = []
            for X, y in dl_val:
                pred = self(X)
                loss.append(self.loss_fn(pred, y).item())
            return np.mean(loss)/num_batches

    def fit(self, dataset_training, num_epochs, num_batches):
        ds_train, ds_val = self._split_train_val(dataset_training, train_rate=0.7)

        for epoch in range(num_epochs):
            loss_train = self._training(ds_train, num_batches)
            loss_val = self._validation(ds_val, num_batches)

            print(f'epoch: {epoch} -> train loss: {loss_train:.5f}, val loss: {loss_val:.5f}')
        
        print(f'\nparameters: {list(self.parameters())}\n')