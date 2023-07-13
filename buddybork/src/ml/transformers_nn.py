"""Deep neural network model for OSR"""

from typing import Optional, Tuple, List, Union

import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.constants_ml import NUM_FEATS_TOT





class TransformerDNN():
    def __init__(self, num_classes: int):
        self._num_classes = num_classes

        self._transform = ToTensor()

    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        # create dataset loader
        batch_size = 64
        dataset = FeatureVectorDataset(X, y, self._transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # training params
        num_epochs = 80
        lr = 0.01
        momentum = 0.5

        scheduler_type = 'exponential'

        scheduler_params = None
        if scheduler_type == 'exponential':
            scheduler_params = dict(gamma=0.80)

        # init, train, test
        self._net, criterion, optimizer, scheduler = init_net(lr, momentum, self._num_classes,
                                                              scheduler_type=scheduler_type,
                                                              scheduler_params=scheduler_params)

        train(dataloader, self._net, criterion, optimizer, num_epochs, scheduler=scheduler)

    def transform(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._net.forward(self._transform(X)).detach().numpy()





class FeatureVectorDataset(Dataset):
    """Feature vector dataset"""
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 transform):
        """
        X and y are the feature array and labels
        """
        self._X = X
        self._y = y
        self._transform = transform

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx: Union[torch.tensor, List[int]]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vec = self._X[idx, :]
        label = self._y[idx]

        vec = self._transform(vec)

        return vec, label


class ToTensor(object):
    """Convert to Tensors"""
    def __call__(self, arr):
        return torch.Tensor(arr)


def init_net(lr: float,
             momentum: float,
             num_classes: int,
             scheduler_type: Optional[str] = None,
             scheduler_params: Optional[dict] = None):
    """Initialize the network and learning modules"""
    net = NN_OSSR(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)

    scheduler = None
    if scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_params['gamma'])

    return net, criterion, optimizer, scheduler


class NN_OSSR(nn.Module):
    """Feedforward neural net for the OSSR system"""
    def __init__(self, num_classes: int):
        super().__init__()

        if 1:
            num_feats_hidden = 64 # lower than this (power of 2) reduces performance
            layers = [
                nn.Linear(NUM_FEATS_TOT, num_feats_hidden),
                nn.ReLU(),
                nn.Linear(num_feats_hidden, num_classes)
            ]

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear_relu_stack(x)
        return logits


def train(trainloader,
          net,
          criterion,
          optimizer,
          num_epochs,
          scheduler = None) \
        -> List[float]:
    """Train"""
    epoch_loss = []
    num_train = len(trainloader) * trainloader.batch_size

    for epoch in range(num_epochs):  # loop over the dataset
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            loss_iter = loss.item()
            running_loss += loss_iter

        epoch_loss.append(running_loss)
        print("epoch %d, loss = %.3f" % (epoch + 1, epoch_loss[epoch]))

        scheduler.step()

    print('Finished Training')

    return epoch_loss