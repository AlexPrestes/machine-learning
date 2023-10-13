import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

class MLP(nn.Module):
    def __init__(self, input_layer=1, hidden_layer=(5, )):
        super().__init__()
        self.activation_fn = nn.ReLU()
        self.loss_fn = nn.MSELoss()
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self._make_mlp()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)

    def forward(self, x):
        y = self.layer(x)
        return y
    
    def _make_mlp(self):
        list_layer = []
        list_layer.append( (self.input_layer, self.hidden_layer[0]) )
        for i in range(len(self.hidden_layer)-1):
            size = (self.hidden_layer[i], self.hidden_layer[i+1])
            list_layer.append(size)

        layers = []
        for l in list_layer:
            layers.append(nn.Linear(*l))
            layers.append(self.activation_fn)
        layers.append( nn.Linear(self.hidden_layer[-1], 1) )

        print(layers)
        self.layer = nn.Sequential( *layers )

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

            print(f'epoch: {epoch+1} -> train loss: {loss_train:.5f}, validate loss: {loss_val:.5f}')
        
        print(f'\nparameters: {list(self.parameters())}\n')