import cv2
import numpy as np
import scipy
import os
import sys
import scipy.io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.feature import greycomatrix, greycoprops
import math
import operator
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors,distances[0][1]
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
DtrainData=scipy.io.loadmat('Ctrainset.mat')
trainData=DtrainData['trainset'].tolist()
Dresponses=scipy.io.loadmat('labels.mat')
responses=Dresponses['labels']
trainingSet=[]
for x in range(len(trainData)):

    
    if responses[x]==1:
        
        trainData[x].append('a')
    else:

        trainData[x].append('b')
        
    trainingSet.append(trainData[x])
TP=0
FP=0
TN=0
FN=0
tp=[]
tn=[]
for  i in range(len(trainingSet)):
    testInstance = trainingSet[i][:]
    if responses[i]==1:
        testInstance.remove('a')
        testcase='a'
    else:
        testInstance.remove('b')
        testcase='b'
        
    k = 1
    neighbors,distances = getNeighbors(trainingSet[:58], testInstance, 1)
    response = getResponse(neighbors)
    
    if response==testcase:
        if response=='a':
            TP+=1
        else:
            TN+=1
        
    else:
        if response=='b':
            FP+=1
        else:
            FN+=1
    tp.append(TP+FP)
    tn.append(TN+FN)
tpr=[float(i)/(TP+FP) for i in tp]
fpr=[float(j)/(TN+FN) for j in tn]
Accuracy=float(TP+TN)/(TP+TN+FP+FN)
print ("--------------------------------------------")
print ("----------------SVM Details-----------------")
print ("Accuracy    =  %3.4f"%Accuracy)

