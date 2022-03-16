#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8

import copy
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class NetApproximator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=32) -> None:
        super(NetApproximator, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        
    def _prepare_data(self, x, requires_grad=False):
        ''' from numpy to tensor '''
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, int):
            x = torch.Tensor([x])
        x.requires_grad = requires_grad
        x = x.float() # from_numpy() produce data in DoubleTensor
        if x.data.dim() == 1:
            x = x.unsqueeze(0)
        return x
    
    def forward(self, x):
        x = self._prepare_data(x)
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred
    
    def __call__(self, x):
        y_pred = self.forward(x)
        return y_pred.data.numpy()
    
    def fit(self, x, y, criterion=None, optimzer=None, epochs=1, learning_rate=1e-4):
        ''' model training '''
        if criterion is None:
            criterion = torch.nn.MSELoss(size_average=False)
        if optimzer is None:
            optimzer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if epochs < 1:
            epochs = 1
        
        y = self._prepare_data(y, requires_grad=False)

        for _ in range(epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred, y)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
        
        return loss
    
    def clone(self):
        return copy.deepcopy(self)
    
    