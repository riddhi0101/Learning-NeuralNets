import prepFunctions
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset
from torch.utils.data import Dataset


def get_same_index(target, label):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)

    return label_indices

# expects mnist dataset
def comparisonDataConsecutive(dataSet):
    indices = []
    #gets all the indices of the data obsv with same y from the dataset that is passed in
    for i in range(5):
        indlist = get_same_index(dataSet.targets, i)
        indices.append(indlist)
    subsets = []
    [subsets.append(torch.utils.data.Subset(dataSet, i)) for i in indices]
    comp = []
    for indi in range(len(indices) - 1):
        comp.append(int(min(len(subsets[indi]), len(subsets[indi + 1]))))
    tot1 = sum(comp)
    x = torch.zeros([tot1, 2, 28, 28], dtype=torch.float32)
    y = torch.zeros([tot1,1])
    # 1 for first pic greater, 0 for first pic less
    k = 0
    for i in range(len(subsets) - 1):
        for j in range(int(comp[i] / 2)):
            x[k][0] = subsets[i][j][0]
            x[k][1] = subsets[i + 1][j][0]
            y[k][0] = 0
            k += 1
        for j in range(int(comp[i] / 2), comp[i]):
            x[k][1] = subsets[i][j][0]
            x[k][0] = subsets[i + 1][j][0]
            y[k][0] = 1
            k += 1
    return x,y


def comparisonDataNonconsecutive(dataSet):
    indices = []
    # gets all the indices of the data obsv with same y from the dataset that is passed in
    for i in range(5):
        indlist = get_same_index(dataSet.targets, i)
        indices.append(indlist)
    subsets = []
    [subsets.append(torch.utils.data.Subset(dataSet, i)) for i in indices]
    comp = {}
    for i in range(len(subsets) - 2):
        for j in range(i + 2, len(subsets), 1):
            comp[(i, j)] = int(min(len(subsets[i]), len(subsets[j])))
    tot = sum(comp.values())
    x = torch.zeros([tot, 2, 28, 28], dtype=torch.float32)
    y = torch.zeros([tot, 1])
    k = 0
    for key, values in comp.items():
        for value in range(int(values / 2)):
            x[k][0] = subsets[key[0]][value][0]
            x[k][1] = subsets[key[1]][value][0]
            y[k][0] = 0
            k += 1
        for value in range(int(values / 2), values):
            x[k][0] = subsets[key[1]][value][0]
            x[k][1] = subsets[key[0]][value][0]
            y[k][0] = 1
            k += 1
    return x,y




class Trainsetcomp(Dataset):
    def __init__(self, x, y):
        self.len = (x.shape[0])
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.len

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

x, y = comparisonDataConsecutive(trainset)
traindataComp = Trainsetcomp(x,y)
x,y = comparisonDataConsecutive(valset)
valdataComp = Trainsetcomp(x,y)


trainloader = torch.utils.data.DataLoader(traindataComp,
                                          batch_size=64,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(valdataComp,
                                          batch_size=64,
                                          shuffle=True)
x,y = comparisonDataNonconsecutive(valset)
testdata = Trainsetcomp(x,y)
testloader = torch.utils.data.DataLoader(testdata,
                                          batch_size=64,
                                          shuffle=True)



#print(traindataComp[0])
#prepFunctions.show_dataComp(valdataComp[0][0], valdataComp[0][1])
#print(len(traindataComp))
#print(traindataComp.y.shape)