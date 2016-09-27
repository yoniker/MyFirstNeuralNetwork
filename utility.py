
import numpy as np
import scipy.misc as smp
import random
import time, sys


"""
showPic(pictures,labels)

given an array of picture (each picture represented by a np array),and an array of labels,
present a random picture and its corresponding label
I put the default picture height and weight as 28 since this is the size of pics of Mnist.

"""

def showPic(pictures,labels,picture_width=28,picture_height=28):
    numPics=len(pictures)
    random.seed(0)
    chosenPicIndex=random.randint(0,numPics)
    img = smp.toimage( pictures[chosenPicIndex].reshape(picture_width,picture_height) )       # Create a PIL image
    print('The label for the pic is:'+str(labels[chosenPicIndex]))
    img.show()    # View in default viewer
    
    
    

    

#credits for the following function - http://stackoverflow.com/questions/3160699/python-progress-bar
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
#input:n  a number between 0-9
#output:a np (10,) array such that the nth place in the vector is one and otherwise the vector is 0s.
def vectorizeDigit(n):
    theVector=np.zeros(10)
    theVector[n]=1
    return theVector