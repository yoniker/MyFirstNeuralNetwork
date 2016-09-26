#a helper py script, which will set up a Neural network and import some libraries for my use,instead of me having to type that in all the time

import network
import numpy as np
import random
net=network.Network([5,3,4,2])
net.show()
i=np.array([1,1])
i=i.reshape(2,1)
y=i*2
i=np.array([1,1,1,1,1])
i=i.reshape(5,1)

sig=network.sigmoid

(deltaB,deltaW)=net.backpropogation(i,y)

random.seed(0)

#x is some ndarray, return the same object modified such that its elements are random int numbers
def randIt(x):

    for i in range(len(x.flat)):
        x.flat[i]=random.randint(1,10)
    return