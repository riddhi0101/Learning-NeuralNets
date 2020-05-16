import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import prepFunctions as pf
from time import time
from torch.utils.data import Dataset

NUM_EPOCHS = 15
LEARNING_RATE = 0.03
BATCH_SIZE = 64
input_size = 784
output_size = 10

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),])

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),])
trainset = dsets.MNIST(root='./../data',
                            train=True,
                            download=True,
                            transform=transform)
valset = dsets.MNIST(root='./../data',
                            train=False,
                            download=True,
                            transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(valset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)


def train(model, criterion, optimizer, epochs=NUM_EPOCHS):
    time0 = time()
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            i += 1
            # Flatten MNIST images into a 784 long vector
            images = images.view(-1, 28 * 28)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
            running_loss += loss.item()
        correct = 0
        for x, y in valloader:
            # validation
            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = 100 * (correct / len(valset))
        useful_stuff['validation_accuracy'].append(accuracy)
        print("Epoch", epoch, 'loss', running_loss / len(trainloader))
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    return useful_stuff

class Modelimg(nn.Module):
    def __init__(self,ind, h1d, h2d, outd):
        super(Modelimg, self).__init__()
        self.lin1 = nn.Linear(ind,h1d)
        self.lin2 = nn.Linear(h1d,h2d)
        self.lin3 = nn.Linear(h2d, outd)
    def forward(self,x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = self.lin3(x)
        return x

input_size = 784
hidden_sizes = [128, 64]
output_size = 10
model = Modelimg(input_size,hidden_sizes[0],hidden_sizes[1],output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
#print((trainset[0][1]))
#print(model(trainset[0][0].view(-1, 28 * 28)).shape)

results = train(model, criterion,optimizer,25)
pf.plot_accuracy_loss(results)
a = model(trainset[0][0].view(-1, 28 * 28))
print(a)
_, yhat = torch.max(a,1)
print(yhat)