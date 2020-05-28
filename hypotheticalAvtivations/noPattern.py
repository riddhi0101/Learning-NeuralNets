import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pairSim as ff

activationsDict = {}
for i in range(4):
    for j in range(i+1,5):
        onestepact = np.random.rand(64)
        compPair = str(i) + str(j)
        activations = ff.addSomeVar(onestepact, .3)
        activationsDict[compPair] = activations
        compPair = str(j) + str(i)
        onestepact = np.random.rand(64)
        activations = ff.addSomeVar(onestepact, .3)
        activationsDict[compPair] = activations

simMatrixm = ff.simMatrix(activationsDict)
axislabels = list(activationsDict.keys())
hm = sns.heatmap(simMatrixm, xticklabels=axislabels, yticklabels=axislabels, cmap="GnBu")
#plt.savefig('hypothetical_dataNoPattern')
plt.show()