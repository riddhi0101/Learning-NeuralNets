import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
activationsDict = {}

def addSomeVar(activations, sigma = 1, mu = 0):

    var = sigma * np.random.standard_normal(activations.shape) + mu
    return activations + var

onestepact = np.random.rand(64)
for i in range(4):
    compPair = str(i) + str(i+1)
    activations = addSomeVar(onestepact, .3)
    activationsDict[compPair] =activations

    compPair = str(i+1) + str(i)
    activations = addSomeVar(onestepact, .3)
    activationsDict[compPair] = activations



twostepact =  np.random.rand(64)
for i in range(3):
    compPair = str(i) + str(i + 2)
    activations = addSomeVar(twostepact, .3)
    activationsDict[compPair] = activations

    compPair = str(i + 2) + str(i)
    activations = addSomeVar(twostepact, .3)
    activationsDict[compPair] = activations
threestepact = np.random.rand(64)
for i in range(2):
    compPair = str(i) + str(i + 3)
    activations = addSomeVar(threestepact, .3)
    activationsDict[compPair] = activations

    compPair = str(i + 3) + str(i)
    activations = addSomeVar(threestepact, .3)
    activationsDict[compPair] = activations

fourstepact = np.random.rand(64)
compPair = str(0) + str(4)
activations = addSomeVar(fourstepact, .3)
activationsDict[compPair] = activations

compPair = str(4) + str(0)
activations = addSomeVar(fourstepact, .3)
activationsDict[compPair] = activations

#print((activationsDict.keys()))
simMatrix = np.zeros((20,20))

for indi, acti in enumerate(activationsDict.values(),start=0):
    for indj, actj in enumerate(activationsDict.values(),start=0):
        simMatrix[indi][indj] = np.dot(acti,actj)/(np.linalg.norm(acti)*np.linalg.norm(actj))
strret = ""

for i in simMatrix:
    newstr = ""
    for j in i:
        newstr += str(j)
        newstr += "\t"
    strret += newstr + "\n"
print(strret)

'''print('here')
simMatrix[0:] = [1]*21
print(simMatrix[0,1:])
print(simMatrix)
print(type(list(activationsDict.keys())))'''
axislabels = list(activationsDict.keys())
hm = sns.heatmap(simMatrix, xticklabels=axislabels, yticklabels=axislabels, cmap="GnBu")
#plt.savefig('hypothetical_dataStep')
plt.show()
#plt.savefig('hypothetical_data')