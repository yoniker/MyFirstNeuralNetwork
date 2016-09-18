"""

a network class. This represnts our Neural network.

a network is represented by the following elements:
1.Its weights
2.Its biases
3.It's architecture.

Given those 3, we can:

1.determine the output for a given output (=> Measure the network various performance metrics)
2.Train the network with a specific training set (backpropogation gardient descent or Stochastic gradient descent )




"""


import numpy as np


class Network:



    """

    architecture is simply list of ints, which will determine the architecture of our network.
    For example, [5,7,2] is a neural network with an input layer of size 5, 7 neurons at the hidden layer, and 2 neurons at the output layer.
    """
    def __init__(self,architecture):
        self.architecture=architecture
        """
        #Weight will be a list of ndarray 2d matrices. Here is how we can construct it:
        Let's say that we have a [2,3,4,5] network. Then the weights matrices will be:
        2*3,3*4,4*5. I achieve that effect by zipping together two lists: [2,3,4] and [3,4,5] eg one without the last neurons layer, and one without the first.
        a (n,m) matrix of random Gaussian values is achieved by the command numpy.random.randn(n,m)
        I will note those lists we iterate over as l-1(lm1) and l
        """
        self.weights=[np.random.randn(lm1,l) for (lm1,l) in zip(architecture[:-1],architecture[1:])]
        """
        Just a small technical detail: we need to transpose each of the weights matrix. Why? well..
        Imagine that we have on layer l-1 4 neurons and on layer l 5 neurons, and that each activations of neurons is represented by a column vector.
        So we need to 'go' from a 4x1 vector to a 5x1 vector. The only way to do that is mul a matrix on the left side like so: (5x4)(4x1).
        The effect could have been achieved by swapping l and l-1 on the code above.
        """
        self.weights=[w.transpose() for w in self.weights]
        self.biases=[np.random.rand(sl,1) for sl in architecture]



    