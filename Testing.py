# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 01:31:49 2020

@author: Home
"""


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*5*5, 256)
        self.fc2 = nn.Linear(256, 10)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.drop(self.pool(x))
        
        x = self.relu(self.conv2(x))
        x = self.drop(self.pool(x))
        
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        
        x = self.relu(self.fc2(x))
        x = self.softmax(x)
        return x
    
model = NeuralNet()

batch_size, C, H, W = 1, 1, 28, 28
x = torch.randn(batch_size, C, H, W)
output = model(x)
