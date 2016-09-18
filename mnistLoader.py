"""
mnistLoader.py
~~~~~~~~~~

A module to implement loading mnist data put out by professor Yann at http://yann.lecun.com/exdb/mnist/
It assumes that the relevant gzip files are already in the working directory, and on top of that it also assumes the following
1.Height,width (and overall image size in pixels) are 28,28 and 28*28 respectively
2.Training dataset contains 60,000 examples, and test dataset contains 10,000 examples
3.Files formats as specified at 18/09/2016 5:31 IST. (eg magic numbers locations,each pixel being an 0-255 integer etc)

"""



import os
import gzip
import numpy as np
import scipy.misc as smp
import random


"""
showPic(pictures,labels)

given an array of picture (each picture represented by a np array),and an array of labels,
present a random picture and its corresponding label


"""

def showPic(pictures,labels):
    numPics=len(pictures)
    chosenPicIndex=random.randint(0,numPics)
    PICTURE_WIDTH=28
    PICTURE_HEIGHT=28
    img = smp.toimage( pictures[chosenPicIndex].reshape(PICTURE_WIDTH,PICTURE_HEIGHT) )       # Create a PIL image
    print('The label for the pic is:'+str(labels[chosenPicIndex]))
    img.show()    # View in default viewer
    
    
    
"""
readImages(file,numberOfPics)
given an open images (test or training) file, formatted as specified at Yann's webpage,
1.Skips the header (format is the same eg the header size in bytes is the same for both test and training files)
2.reads #numberOfPics pictures from that file (no error handling whatsoever,function is not meant for usage as a generic function, but at the specific peculiar format Yann provided)
3.closes the file
4.returns the images as a np array.
"""
def readImages(file,numberOfPics):
    HEADER_SIZE_IN_BYTES=16
    PICTURE_WIDTH=28
    PICTURE_HEIGHT=28
    PICTURE_SIZE_IN_PIXELS=PICTURE_HEIGHT*PICTURE_WIDTH
    HEADER_SIZE_IN_BYTES=16
    #skip the header
    file.read(HEADER_SIZE_IN_BYTES)
    images=[]
    for i in range(numberOfPics):
        currentPicInBytes=file.read(PICTURE_SIZE_IN_PIXELS) #This reads the picture in str format such as '\x00\x00\x00\x00'
        picture=[ord(n) for n in currentPicInBytes] #This translates the string to numbers
        picInNp=np.asarray(picture,dtype=float)
        #If I want it in a viewable format: picInNp=picInNp.reshape(PICTURE_HEIGHT,PICTURE_WIDTH)
        picInNp=picInNp/255
        images.append(picInNp)
    file.close()
    images=np.asarray(images)
    return images
        

"""
same function as readImages, only that now the labels are being read.


"""
def readLabels(file,numberOfLabels):
    HEADER_SIZE_IN_BYTES_TRAINING_LABELS=8
    file.read(HEADER_SIZE_IN_BYTES_TRAINING_LABELS) #skip header
    allLabelsInString=file.read(numberOfLabels)
    labels=[ord(n) for n in allLabelsInString]
    file.close()
    labels=np.asarray(labels)
    return labels
    
    
"""
readData()

Read the files Yann has provided and returns a tuple (training_data,valudation_data,test_data)
Each of the _data elements  is a tuple (x,y) by itself, x being the images ndarray,y being the labels ndarray
"""
def readData():
    
    f=gzip.open('train-images-idx3-ubyte.gz','rb')
    trainImages=readImages(f,60000)
    #Now read training labels
    f=gzip.open('train-labels-idx1-ubyte.gz','rb')
    trainLabels=readLabels(f,60000)
    f=gzip.open('t10k-images-idx3-ubyte.gz','rb')
    testImages=readImages(f,10000)
    f=gzip.open('t10k-labels-idx1-ubyte.gz','rb')
    testLabels=readLabels(f,10000)
    #Alright, now let's separate training data into 'training' (50k examples) and cv (10k)
    validImages=trainImages[50000:,:]
    validLabels=trainLabels[50000:]
    #And now let's 'cut' the training data
    trainImages=trainImages[:50000,:]
    trainLabels=trainLabels[:50000]
    
    training_data=(trainImages,trainLabels)
    validation_data=(validImages,validLabels)
    test_data=(testImages,testLabels)
    
    
    return (training_data, validation_data, test_data)
    



    

    




    

