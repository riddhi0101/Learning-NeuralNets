import numpy as np

layersd = [4,3,2,1]
weights = []
biases = []

for dim in range(len(layersd) - 1):
    weights.append(np.random.rand(layersd[dim + 1], layersd[dim]))

for dim in range(1, len(layersd)):
    biases.append(np.random.rand(layersd[dim], 1))

for i in zip(weights,biases):
    print(i[0].shape)
    print(i[1].shape)
