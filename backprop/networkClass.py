import numpy as np
import random

class Networkmy():

    def __init__(self, layersd):
        #layersd is a list of the number of neurons in each layer- the first is the input dimension
        self.layersD = layersd
        self.weights = []
        self.biases = []
        self.layersNum = len(layersd)

        for dim in range(self.layersNum - 1):
            self.weights.append(np.random.rand(layersd[dim+1],layersd[dim]))

        for dim in range(1,self.layersNum):
            self.biases.append(np.random.rand(layersd[dim],1))

    def forward(self,x):
        for i in zip(self.weights, self.biases):
            x = np.matmul(i[0],x) + i[1]
        return x

    #data is a list of tuples
    def train(self,data,batchsize,epochs):

        for i in range(epochs):
            batches = self.slpitBatches(data,batchsize)
            for miniBatch in batches:
                # TODO Backpropogation- calc gradients, then update the weights


    def slpitBatches(self,data,batchsize):
        #random.shuffle(data)
        rep = len(data)/batchsize
        newL = []
        for i in range (0, len(data), batchsize):
            newL.append(data[i:i+batchsize])
        return newL







test = Networkmy([4,3,4,2])
'''print('biases')
for i in test.biases:
    print(i.shape)
print('weights')
for i in test.weights:
    print(i.shape)
x = np.array([[0],[1],[2],[3]])
fp = test.forward(x)
#print(fp)'''

data = []
for i in range(50):
    data.append((i,i))
splitData = test.slpitBatches(data,5)
print(splitData)
a = np.array(splitData)
print(a.shape)


