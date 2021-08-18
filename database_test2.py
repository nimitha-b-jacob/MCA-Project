import os
import glob
import cv2
import numpy as np
import math
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import hog
import pickle
from skimage import img_as_ubyte
from tkFileDialog import askopenfilename
from tkFileDialog import *
index = {}
images = {}
key=1
index_test={}
counter=1
filecount=0
for imagePath in os.listdir("C://Users//nimit//Documents//wood defect//Nimitha//database test2//all//"):
    filename = imagePath[imagePath.rfind("/") + 1:]
    filecount+=1
for ifile in range(filecount):
    
    imagefile='C://Users//nimit//Documents//wood defect//Nimitha//database test2//all//clearwood_256_'+str(ifile+1)+'.jpg'
    image=cv2.imread(imagefile)
    image=cv2.resize(image,(64,64))
    Gimage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    hist=cv2.calcHist([image], [0, 1, 2], None, [8, 2, 2],
		[0, 256, 0, 256, 0, 256])
    
    hist = cv2.normalize(hist).flatten()
    

    glcm = greycomatrix(Gimage, [5], [0], 256, symmetric=True, normed=True)
    dissimilarity=(greycoprops(glcm, 'dissimilarity')[0, 0])
    correlation=(greycoprops(glcm, 'correlation')[0, 0])
    homogeneity=(greycoprops(glcm, 'homogeneity')[0, 0])
    contrast=(greycoprops(glcm, 'contrast')[0, 0])
    energy=(greycoprops(glcm, 'energy')[0, 0])

    fd, hog_image = hog(Gimage, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    

    feat=[]
    for i in range(len(hist)):
        feat.append(hist[i])
    for i in range(len(fd)):
        feat.append(fd[i])
    feat.append(dissimilarity)
    feat.append(correlation)
    feat.append(homogeneity)
    feat.append(contrast)
    feat.append(energy)
    feature=np.asarray(feat)
    index[str(key)] = feature
    key=key+1
pickle.dump( index, open( "wooddatatest.p", "wb" ) )


