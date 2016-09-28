#a helper py script, which will set up a Neural network and import some libraries for my use,instead of me having to type that in all the time

import network
import utility
import numpy as np
import random
import mnistLoader as loader
net=network.Network([784,40,10])

#First of all,load all the data...
print('Loading the Data....')
(training_data, validation_data, test_data)=loader.readData()
training_examples=training_data[0]
training_labels=training_data[1]
m=training_examples.shape[1]

training_data=[(training_examples[:,i],training_labels[:,i]) for i in range(m)]
    
net.stochastic_grad(training_data,learningRate=3,epochs=10,batchSize=30,test_data=test_data)

_,correct,wrong=net.evaluate(test_data)

#input: a list of tuples (example,label) where we were wrong.
#output: show a random wrong example,with the system's guess.
def show_wrong_example(wrong):
    (examples,labels)=utility.separateListOfTuples(wrong)
    utility.showPic(examples,labels,net)
    