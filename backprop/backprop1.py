import numpy as np
import  math
### forward and backward pass with numpy


## neural net with hidden layers
d_in = 2
d_out = 2
h1 = 4
h2 = 6
a = np.array([[1],[2]])
goal = np.array([[24],[25]])

'''w1 = np.random.randn(h1,d_in)
w2 = np.random.rand(h2,h1)
w3 = np.random.rand(d_out,h2)

b1 = np.random.randn(h1,1)
b2 = np.random.randn(h2,1)
b3 = np.random.randn(d_out,1)'''

w1s = np.array([[3,4],[2,1],[2,5],[0,3]])

w2s = np.array([[1,0,0,2],[2,1,0,0]])
b1s= np.array ([[1],[1],[0],[0]])
b2s = np.array([[1],[0]])

def forward(a):
    actL1 = np.dot(w1s,a)
    actL2 = np.dot(w2s,actL1)
    #actL3 = np.dot(w3,actL2) + b3
    return actL2


yhat = forward(a)

loss = np.square(goal - yhat).sum()
print(loss)


