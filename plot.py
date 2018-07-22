import numpy as np
import matplotlib.pyplot as plt
import sys

result = np.load('result.npy')
output = result[:,0:10]
label = result[:,10].astype(int)
pred = result[:,11].astype(int)
X = []
Y = []

for i in range(10):
    right = label==i
    h = plt.hist([ output[right,i], output[~right,i] ],
                 bins=int(sys.argv[1]),
                 histtype="barstacked")
    x = h[1][0:int(sys.argv[1])]
    y = h[0][0] / h[0][1]
    X.append(x)
    Y.append(y)
X = np.array(X)
Y = np.array(Y)
fig = plt.figure()
for i in range(10):
    plt.plot(X[i],Y[i], label='{}'.format(i))
    plt.legend(loc='upper left')
plt.savefig('plot.png')
