
import numpy as np
import scipy.misc as smp
import random


"""
showPic(pictures,labels)

given an array of picture (each picture represented by a np array),and an array of labels,
present a random picture and its corresponding label
I put the default picture height and weight as 28 since this is the size of pics of Mnist.

"""

def showPic(pictures,labels,picture_width=28,picture_height=28):
    numPics=len(pictures)
    chosenPicIndex=random.randint(0,numPics)
    img = smp.toimage( pictures[chosenPicIndex].reshape(picture_width,picture_height) )       # Create a PIL image
    print('The label for the pic is:'+str(labels[chosenPicIndex]))
    img.show()    # View in default viewer