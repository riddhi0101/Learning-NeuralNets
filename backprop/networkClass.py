import numpy as np
import random
# TODO Backpropogation- calc gradients, then update the weights
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
    def train(self,data,batchsize,epochs,lr):

        for i in range(epochs):
            batches = self.slpitBatches(data,batchsize)
            for miniBatch in batches:
                self.updateMiniBatch(miniBatch,lr)

    def updateMiniBatch(self,minibatch, lr):
        for x,y in minibatch:
            gradW, gradB = self.backprop(x,y)
            #update it
            for i in range(self.layersNum-1):
                self.weights[i] = self.weights[i] - (lr/len(minibatch)) * gradW[i]
                self.biases[i] = self.biases[i] - (lr/len(minibatch)) * gradB[i]

    def backprop(self,x,y):
        gradB = [np.zeros(b.shape) for b in self.biases]
        gradW = [np.zeros(w.shape) for w in self.weights]

        z = x
        zvals = [x]
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,z) + b
            zvals.append(z)

        delta = (zvals[-1] - y)
        gradB[-1] = delta
        gradW[-1] = np.dot(delta, zvals[-2].transpose())

        for l in range(-2, -1*self.layersNum,-1):
            z = zvals[l]
            delta = np.dot(self.weights[l+1].T,delta)
            gradB[l] = delta
            gradW[l] = np.dot(delta, zvals[l-1].transpose())

        return gradW, gradB










    def slpitBatches(self,data,batchsize):
        #random.shuffle(data)
        rep = len(data)/batchsize
        newL = []
        for i in range (0, len(data), batchsize):
            newL.append(data[i:i+batchsize])
        return newL







test = Networkmy([4,3,4,2])
x = np.array([[0],[1],[2],[3]])
y = np.array([[1],[0]])
data = [(x,y)]
for i,j in zip(test.weights,test.biases):
    print(i)
    print(j)
test.train(data,1,1,.3)
print('after training')
for i,j in zip(test.weights,test.biases):
    print(i)
    print(j)

'''print('biases')
for i in test.biases:
    print(i.shape)
print('weights')
for i in test.weights:
    print(i.shape)
x = np.array([[0],[1],[2],[3]])
fp = test.forward(x)
print(fp)

data = []
for i in range(50):
    data.append((i,i))
splitData = test.slpitBatches(data,5)
print(splitData)
a = np.array(splitData)
print(a.shape)'''


