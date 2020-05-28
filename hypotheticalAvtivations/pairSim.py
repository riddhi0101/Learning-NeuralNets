import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

#activationsDict = {}

def addSomeVar(activations, sigma = 1, mu = 0):

    var = sigma * np.random.standard_normal(activations.shape) + mu
    return activations + var

def simMatrix(activationsDict):
    simMatrixret = np.zeros((len(activationsDict.keys()),len(activationsDict.keys())))
    for indi, acti in enumerate(activationsDict.values(), start=0):
        for indj, actj in enumerate(activationsDict.values(), start=0):
            simMatrixret[indi][indj] = np.dot(acti, actj) / (np.linalg.norm(acti) * np.linalg.norm(actj))
    return simMatrixret

'''for i in range(4):
    for j in range(i+1,5):
        onestepact = np.random.rand(64)
        compPair = str(i) + str(j)
        activations = addSomeVar(onestepact, .3)
        activationsDict[compPair] = activations
        compPair = str(j) + str(i)
        activations = addSomeVar(onestepact, .3)
        activationsDict[compPair] = activations
'''
'''print((len(activationsDict.keys())))
simMatrixm = simMatrix(activationsDict)
axislabels = list(activationsDict.keys())
hm = sns.heatmap(simMatrixm, xticklabels=axislabels, yticklabels=axislabels, cmap="GnBu")
plt.savefig('hypothetical_dataPairs')
plt.show()'''
