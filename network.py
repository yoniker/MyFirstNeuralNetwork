"""

a network class. This represnts our Neural network.

a network is represented by the following elements:
1.Its weights
2.Its biases
3.Its architecture.

Given those 3, we can:

1.determine the output for a given output (=> Measure the network various performance metrics)
2.Train the network with a specific training set (backpropogation gardient descent or Stochastic gradient descent )




"""


import numpy as np
import random

import utility


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
        self.biases=np.array([np.random.rand(sl,1) for sl in architecture[1:]]) #input layer doesn't have a bias.
    """ 
    feedForward- simply calculate the net output,given a specific input
    TO DO: Add a parameter which will determine the activation function (and probably default it to sigmoid)
    """
    def feedForward(self,x):
        a=x
        for (w,b) in zip(self.weights,self.biases):
            a=sigmoid(np.dot(w, a)+b)
        return a
        
        
    #Backpropagation algorithm for calculating dC/dw and dC/db. x is the input and y is the expected output.
    #The output will be dC/dW and dC/dB, which are lists of the same size as the network weights and biases respectively.
    def backpropogation(self,x,y):
        #first I will feed forward the input x,calculating everything which needs to be calculated along its path
        activations=[x] #This will be a list of column vectors with the network's activations.
        Zs=[] #Likewise,Zs will be a list of the z's eg w*a+b before applying the activation function
        currentLayerOutput=x
        for(w,b) in zip(self.weights,self.biases):
            newZ=np.dot(w,currentLayerOutput)+b
            Zs.append(newZ)
            currentLayerOutput=sigmoid(newZ)
            activations.append(currentLayerOutput)
            
        #Let's compute delta now. delta will be a list of the deltas for the various layers
        delta=[]
        #According to BP's first formula, deltaL=grad(C,a)(*)sigmoid_prime(ZL) Where L is the last layer of the network, and (*) is element wise multiplication of the two vectors (Hadamard's product)
        cost_derivitives=cost_prime(np.asarray(activations[-1]),np.asarray(y))
        zL=Zs[-1]
        delta.append(np.multiply(cost_derivitives,sigmoid_prime(zL))) #np.multiply is the element wise multiplication (aka Hadamard's product).
        #and now the actual backpropogation part begins, according to equation BP2 deltal=(w[l+1]'*delta[l+1])(*)sigmoid_prime(z[l])
        numberOfLayers=len(self.architecture)
        for l in range(numberOfLayers-1,1,-1): #Not including layer 1 since delta1 doesnt make sense (an error on the input?! :) )
             #this.weights[0] is the same as the book's weight two (transfer between layer 1 and layer 2 is W2 so therefore the indexing)
            wTimesDelta=np.dot(self.weights[l-1].transpose(),delta[0])  #w[l-1] is in fact the book's w[l+1],delta[0] is the last delta we computed
            delta=[np.multiply(wTimesDelta,sigmoid_prime(Zs[l-2]))]+delta #same indexing story for Zs,the book's Z[2] is actually Z[0] here
        #Alright,so far we have calculated delta (which is by the way dC/db.) Now we want to calculate the matrix dC/dW.
        #This is done using BP4 eg ain*deltaout. I will call dC/dw deltaW.
        deltaW=[]
        for i in range(len(delta)):
            deltaW.append(np.dot(delta[i],activations[i].transpose()))
        return (delta,deltaW) #again, remember that according to BP3 delta is dC/dB...
        
        
        
        
    """

    Stochastic gradient descent.
    Input:
    
    training_data - a list of tuples (example,label) 

    """
    
    def stochastic_grad(self,training_data,learningRate,epochs=50,batchSize=0):
    #So basically stochastic grad is the same as gradient descent when batch size is same as the size of our learning set.
    #That will be the default value then.
        if batchSize==0:
            batchSize=len(training_data)
        random.seed(0)
        random.shuffle(training_data)
        training_examples=[]
        training_labels=[]
        training_examples[:],training_labels[:]=zip(*training_data)     
        utility.showPic(training_examples,training_labels)
        return
    
    
        
    
    
    
        
            
            
                   
        
    def show(self):
        i=0
        for w in self.weights:
            i=i+1
            print(("w[{}]="+str(w)) .format(str(i)))
        i=0
        for b in self.biases:
            i=i+1
            print(('b[{}]='+str(b)) .format(str(i)))
            
            #Set the weights and biases
    def set(self):
        layer=0
        for w in self.weights:
            layer=layer+1
            for i in range(0,len(w.flat)):
                w.flat[i]=(layer+1)*((-1)**layer)
        layer=0
        for b in self.biases:
            layer=layer+1
            for i in range(0,len(b.flat)):
                b.flat[i]=layer
            
            
            
        
    
    
    
    
    
    
def cost_prime(output,y):
    return output-y.reshape(output.shape)

    
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

    

#Input : delta- a list containing the error vectors. Notice that delta[0] is delta2 ie the error in Layer2 (there is no point talking about delta1 ie error in the input layer)    
def calcWeights(activations,delta):
    weights=[]
    for i in range(len(delta)):
        weights.append(np.dot(delta[i],activations[i].transpose()))
    return weights
    
    
    
        
    

    