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

ind = 2 * 28 * 28
hiddendim = [256,128,64]
outd = 1

class ModelFull(nn.Module):
    def __init__(self, ind,h1d,h2d,h3d,outd):
        super(ModelFull, self).__init__()
        self.lin1 = nn.Linear(ind, h1d)
        self.lin2 = nn.Linear(h1d, h2d)
        self.lin3 = nn.Linear(h2d, h3d)
        self.lin4 = nn.Linear(h3d, outd)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = torch.sigmoid(self.lin4(x))
        return x

def train(model, criterion, optimizer, epochs = 30):
    lossList = []
    #time0 = time()
    for i in range(epochs):
        runningLoss = 0
        for x, y in dataPairing.trainloader:
            optimizer.zero_grad()
            yhat = model(x.view(-1, 2 * 28 *28))
            #print(yhat.shape)
            #print(y.shape)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
        print('epoch ', i, ' loss: ', str(runningLoss / len(dataPairing.traindataComp)))
        lossList.append(runningLoss / len(dataPairing.traindataComp))

model = ModelFull(ind,hiddendim[0],hiddendim[1],hiddendim[2], outd)
#a = model(dataPairing.traindataComp[0][0].view(-1,2*28*28))
#print(a)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
results = train(model,criterion,optimizer,30)

totcount = 0
correctcount = 0
for x,y in dataPairing.valloader:
    x = x.view(-1, 2 * 28 *28)
    with torch.no_grad():
        yhat = model(x)
    ones = torch.ones(yhat.shape)
    yhat = torch.where(yhat>.9, ones, yhat)
    z = torch.zeros(yhat.shape)
    yhat = torch.where(yhat<0.1, z, yhat)
    for i,j in zip(yhat,y):
        if i[0] == j[0]:
            correctcount+=1
        totcount+=1
print(correctcount)
print(totcount)
#print(len(dataPairing.valdataComp))
print('valset accuracy: ', correctcount/totcount)

for x,y in dataPairing.testloader:
    x = x.view(-1, 2 * 28 *28)
    with torch.no_grad():
        yhat = model(x)
    ones = torch.ones(yhat.shape)
    yhat = torch.where(yhat>.9, ones, yhat)
    z = torch.zeros(yhat.shape)
    yhat = torch.where(yhat<0.1, z, yhat)
    for i,j in zip(yhat,y):
        if i[0] == j[0]:
            correctcount+=1
        totcount+=1
print(correctcount)
print(totcount)
print('test set accuracy: ', correctcount/totcount)


