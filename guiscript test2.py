from Tkinter import *
from time import *
import tkMessageBox
import tkFont
from PIL import ImageTk,Image
import Tkinter
import os
import glob
import cv2
import numpy as np
import math
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import hog
import pickle
from skimage import img_as_ubyte
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import *
size1 = 1500,700

class app:
    def click(self):
    
        self.Message.config(text = "THANK YOU")
        self.mas.update()
        sleep(1)
        self.mas.destroy()
    
    def __init__(self):
        
        self.mas=Tk()
        self.image1=Image.open('Needwood-Forest.jpg')
        self.image1=self.image1.resize(size1)
        self.background=ImageTk.PhotoImage(self.image1)
    
        self.img_label = Label(self.mas, image=self.background)
        self.img_label.pack()

        self.mas.title("Wood Defect Detection")
        self.title = Label(self.mas,text = "Wood Defect Detection",width=20)
        self.mas.minsize(width=1500,height = 700)
        self.mas.maxsize(width=1500,height = 700)

        
        self.title.config(font = ("Lucida Calligraphy",22) ,fg = "green",bg = "white")
        self.title.place(x=500,y=15)

        self.Message = Label(self.mas,text = "MESSAGE",width=32,height=2,wraplength = 300)
        self.Message.config(font = ("Times New Roman",12),bg="white")
        self.Message.place(x=560,y=90)
        
        self.image = Frame(self.mas,width=150,height=150)
        self.image.pack_propagate(0)
        self.image.place(x=900,y=240)

        self.im = Label(self.image,text = "Image",width=100,height=150)
        self.im.config(bg = "White")
        self.im.pack(side = "top")

        self.test = Button(self.mas,text = "Test",command = self.woodtest,width="17",height="4")
        self.test.config(font = ("Times New Roman",13),bg="white")
        self.test.place(x=350,y = 270)

        self.exit = Button(self.mas,text = "QUIT",command = self.click,width="17",height="2")
        self.exit.config(font = ("Times New Roman",14),bg="white",fg = "red")
        self.exit.place(x=600,y=470)
        
        self.mas.mainloop()

    
            
   

    def woodtest(self):

        result ,image = test_wood().wood()
        
        if (result==2.0):

            self.Message.config(text = "The wood has knots")
            self.temp = Image.open(image)
            self.temp = self.temp.resize((150,150))
            self.t = ImageTk.PhotoImage(self.temp)
            self.image = Tkinter.Frame(self.mas,width=150,height=150)
            self.image.pack_propagate(0)
            self.image.place(x=900,y=240)
            self.im = Tkinter.Label(self.image,image = self.t,width=150,height=150)
            self.im.pack(side = "top")

            self.mas.update()
        if (result==1.0):
 
            self.Message.config(text = "the wood hs cracks")
            self.temp = Image.open(image)
            self.temp = self.temp.resize((150,150))
            self.t = ImageTk.PhotoImage(self.temp)
            self.image = Tkinter.Frame(self.mas,width=150,height=150)
            self.image.pack_propagate(0)
            self.image.place(x=900,y=240)
            self.im = Tkinter.Label(self.image,image = self.t,width=150,height=150)
            self.im.pack(side = "top")
            
            self.mas.update()
        if (result==0.0):
            self.Message.config(text = "the wood is clear")
            self.temp = Image.open(image)
            self.temp = self.temp.resize((150,150))
            self.t = ImageTk.PhotoImage(self.temp)
            self.image = Tkinter.Frame(self.mas,width=150,height=150)
            self.image.pack_propagate(0)
            self.image.place(x=900,y=240)
            self.im = Tkinter.Label(self.image,image = self.t,width=150,height=150)
            self.im.pack(side = "top")
            self.mas.update()


class test_wood:
    
    def wood(self):
        
        images={}
        index_test={}
        results={}
        reverse=True
        train=np.zeros((600,1),np.float32)
        featureset=np.zeros((1,165),np.float32)
        dataset=np.zeros((600,165),np.float32)
        svm_params = dict( kernel_type = cv2.SVM_RBF,svm_type=cv2.SVM_C_SVC,C=32.67,gamma=0.2 )
        
        for g in range(600):
            if(g>201) and (g<400):
                train[g,0]=1
            if(g>400) and (g<600):
                train[g,0]=2
        Tk().withdraw()
        filename = askopenfilename()
        img_= cv2.imread(filename)
        disp = img_.copy()
        grey = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        value = (5,5)
        blurred = cv2.GaussianBlur(grey, value, 0)
        _, thresh1 = cv2.threshold(blurred, 127, 255,
                                       cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
                                               cv2.CHAIN_APPROX_NONE)
        contours1, hierarchy1 = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
                                               cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(disp,contours1,-1,(0,255,0),1)
        max_area = -1
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
        cnt=contours[ci]
        cv2.drawContours(img_,[cnt],0,(0,255,0),1)

        image=cv2.resize(img_,(64,64))
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
        feature1=np.asarray(feat)
        feature1 = np.asarray(feature1 ,np.float32)
        index = pickle.load( open( "wooddatatest.p", "rb" ) )
        i=0
        for (k, feature) in index.items():
            dataset[i,:] = feature
            i+=1
        featureset[0,:]=feature1
        svm=cv2.SVM()
        svm.train(dataset,train,params=svm_params)
        result=svm.predict(featureset)
        print (result)
        if (result==2.0):
            cv2.imshow('AllContours',disp)
            # print 'The wood has knots:\n The solution is to fill the wood knots and voids with epoxy'
            print ('the wood has knots')
            return result,filename
        if (result==1.0):
            cv2.imshow('AllContours',disp)
            # print 'thw wood has crack: Cracks in wooden structures can be filled with adhesives or with thermoplastic composite material.'
            print ('the wood has cracks')
            return result,filename
        if (result==0.0):
            print ('The wood is clear')
            return result,filename

if __name__=="__main__":
   
    ap=app()
