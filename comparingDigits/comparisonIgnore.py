import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
from time import time
from torch.utils.data import Dataset
import dataPairing
import prepFunctions as pf


input_size = 784
hidden_sizes = [128, 64]
output_size = 10

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.net1lin1 = nn.Linear(input_size, hidden_sizes[0])
        self.net1lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.net1lin3 = nn.Linear(hidden_sizes[1], output_size)
        self.net2lin1 = nn.Linear(input_size, hidden_sizes[0])
        self.net2lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.net2lin3 = nn.Linear(hidden_sizes[1], output_size)
        self.net12lin1 = nn.Linear(2, 4)
        self.net12lin2 = nn.Linear(4, 1)

    def forward(self,x):
        x1 = x[0].view(-1, 28 * 28)
        x2 = x[1].view(-1, 28 * 28)
        x1 = torch.relu(self.net1lin1(x1))
        x1 = torch.relu(self.net1lin2(x1))
        x1 = self.net1lin3(x1)
        x2 = torch.relu(self.net2lin1(x2))
        x2 = torch.relu(self.net2lin2(x2))
        x2 = self.net2lin3(x2)

        _, img1 = torch.max(x1, 1)
        _, img2 = torch.max(x2, 1)
        img1 = img1[0]
        img2 = img2[0]
        net2in = torch.tensor([img1,img2])
        x3 = torch.relu(self.net12lin1(net2in))
        x3 = torch.sigmoid(self.net12lin2(x3))
        return x3

def train(model,criterion, optimizer, epochs = 20):
    #useful_stuff = {'training_loss': [], 'validation_accuracy': []}
    lossList = []
    time0 = time()
    for i in range(epochs):
        runningLoss = 0
        for x,y in dataPairing.traindataComp:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            runningLoss+= loss.item()
        print('epoch ', i, ' loss: ', str(runningLoss/len(dataPairing.traindataComp)))
        lossList.append(runningLoss / len(dataPairing.traindataComp))

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
## dont think this will work because i dont think it will propogate through the classification networks
result = train(model, criterion, optimizer)


