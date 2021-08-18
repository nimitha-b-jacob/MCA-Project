import numpy as np
import cv2
import pickle
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import hog
import glob
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import *
images={}
index_test={}
results={}
reverse=True
train=np.zeros((510,1),np.float32)
##print train
featureset=np.zeros((1,165),np.float32)
dataset=np.zeros((510,165),np.float32)
svm_params = dict( kernel_type = cv2.SVM_RBF,svm_type=cv2.SVM_C_SVC,C=2.67, gamma=5.383 )
for g in range(510):
    if(g>102) and (g<174):
        train[g,0]=1
    if(g>173) and (g<301):
        train[g,0]=2
    if(g>300):
        train[g,0]=3
##print train
#from matplotlib import pyplot as plt
Tk().withdraw()
filename = askopenfilename()
##for i in range(1,511):
####filename = "D:/cpy/projects2019/Fuzzy/plant leaf deseases/Rose/leaf/"+str(i)+".JPG"
##print filename
img_= cv2.imread(filename)
grey = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
##    cv2.imshow('gray',grey)
##    cv2.waitKey()
value = (5,5)
blurred = cv2.GaussianBlur(grey, value, 0)
##    cv2.imshow('blurred',blurred)
##    cv2.waitKey()
_, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
##    cv2.imshow('thresh',thresh1)
##    cv2.waitKey()
contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
                                       cv2.CHAIN_APPROX_NONE)
##contours1, hierarchy1 = cv2.findContours(thresh1[38:76,0:190].copy(),cv2.RETR_TREE, \
##                                       cv2.CHAIN_APPROX_NONE)
max_area = -1
for i in range(len(contours)):
    cnt=contours[i]
    area = cv2.contourArea(cnt)
    if(area>max_area):
        max_area=area
        ci=i
cnt=contours[ci]
##    cv2.drawContours(img_,[cnt],0,(0,255,0),1)
##    cv2.imshow('contours',img_)
##    cv2.waitKey()
##    cv2.destroyAllWindows()
x,y,w,h = cv2.boundingRect(cnt)
mask=(thresh1[y:y+h,x:x+w])
imag_=cv2.bitwise_and(img_[y:y+h,x:x+w],img_[y:y+h,x:x+w],mask=mask)
for i_imag in range(imag_.shape[0]):
    for j_imag in range(imag_.shape[1]):
        if imag_[i_imag,j_imag,1]>110:
            imag_[i_imag,j_imag,:]=[0,0,0]
filename_='test'+'.jpg'
cv2.imwrite(filename_,imag_)

imag_=cv2.resize(imag_,(64,64))
image=imag_
##    image1 = cv2.imread(imagePath,1)
##    image=cv2.resize(image1,(36,128))
#image=cv2.imread('/root/Desktop/arun.jpg',1)
Himage=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
Gimage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
###HSV Histogram

hist=cv2.calcHist([image], [0, 1, 2], None, [8, 2, 2],
            [0, 256, 0, 256, 0, 256])
hist = cv2.normalize(hist).flatten()

##GLCM features
glcm = greycomatrix(Gimage, [5], [0], 256, symmetric=True, normed=True)
dissimilarity=(greycoprops(glcm, 'dissimilarity')[0, 0])
correlation=(greycoprops(glcm, 'correlation')[0, 0])
homogeneity=(greycoprops(glcm, 'homogeneity')[0, 0])
contrast=(greycoprops(glcm, 'contrast')[0, 0])
energy=(greycoprops(glcm, 'energy')[0, 0])

##Edge Histogram of Oriented Gradients
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
feature1=np.asarray(feat)

##
##
##index_test["8"]=feature1
index = pickle.load( open( "data.p", "rb" ) )
###bf = cv2.BFMatcher()
i=0
for (k, feature) in index.items():
    # compute the distance between the two histograms
    # using the method and update the results dictionary
##    print k
    dataset[i,:] = feature
    i+=1
featureset[0,:]=feature1
svm=cv2.SVM()
svm.train(dataset,train,params=svm_params)
result=svm.predict(featureset)
print result
if (result==3.0):
    print 'The leaf is not diseased'
if (result==2.0):
    print 'The leaf is caused by powdery mildew'
if (result==1.0):
    print 'The leaf is caused by downy mildew'
if (result==0.0):
    print 'The leaf is caused by black spot fungus'

