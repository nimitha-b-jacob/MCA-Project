from flask import Flask
import os
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import glob
import cv2
import numpy as np
import math
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import hog
import pickle
from skimage import img_as_ubyte
size1 = 1500,700
app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','PNG'}


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_PATH']='static/upload'
@app.route('/')
def crpti():
    return render_template('index.html')

@app.route('/upload')
def wood_dt():
    return render_template('detection.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_image():
    
    if request.method == 'POST':
        print("Hai")
        f = request.files['form1']
        
        f.save(os.path.join(app.config['UPLOAD_PATH'],f.filename))
        loc='static/upload/'+f.filename
        print(loc)
        result,img = wood(loc)
        if (result==2.0):
            cond = "The wood has knots:"
            return render_template('display.html',image1=loc,image2=img,wood_type=cond)
            
        if (result==1.0):
            cond = "The wood has crack:"
            return render_template('display.html',image1=loc,image2=img,wood_type=cond)

        if (result==0.0):
            cond = "The wood is clear"
            return render_template('uploaded.html',image1=loc,wood_type=cond)           
        return render_template('uploaded.html')
    return render_template('uploaded.html')


def wood(path):
           
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
    filename = path
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
    cv2.imwrite('static/upload/draw/1.jpg',disp)
    path='static/upload/draw/1.jpg'
    return result,path


if __name__=="__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)


