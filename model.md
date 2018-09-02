# OCR
OCR for handwriting recognition

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 00:14:57 2018

@author: Anchal
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D


def main():
    path='H:\Data'
    path1='H:\Data_1'
    
    imlist=os.listdir(path)
    for file in imlist:
        im=cv2.imread(path+'\\'+file)
        im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(im,(200,200))
        cv2.imwrite(path1+'\\'+file,img)
       
    imlist=os.listdir(path1)
    
    for i in range(0,571):
        im1=cv2.cvtColor((cv2.imread(path1+'\\'+imlist[i])), cv2.COLOR_BGR2GRAY)        
        
    #m,n,o=imatrix.shape[0:2]
    #print(m)
    #print(n)    
    
    n_samples=size(imlist)
    print ('Number of samples:',n_samples)
    
    print('Each image resized to:200 x 200')
    imatrix=array([array(cv2.cvtColor((cv2.imread(path1+'\\'+im2)), cv2.COLOR_BGR2GRAY)).flatten()
                for im2 in imlist],'f')
    m,n=imatrix.shape[0:2]
    print('Matrix created of dimensions:',m,'x',n)
    print('Number of labels: 26')
    
    label=np.ones((n_samples,),dtype=int)
    label[0:2017]=0
    label[2017:4033]=1
    label[4033:6049]=2
    label[6049:8065]=3
    label[8065:10081]=4
    label[10081:12097]=5
    label[12097:14113]=6
    label[14113:16129]=7
    label[16129:18145]=8
    label[18145:20161]=9
    label[20161:22177]=10
    label[22177:24193]=11
    label[24193:26209]=12
    label[26209:28225]=13
    label[28225:30241]=14
    label[30241:32257]=15
    label[32257:34273]=16
    label[34273:36289]=17
    label[36289:38305]=18
    label[38305:40321]=19
    label[40321:42337]=20
    label[42337:44353]=21
    label[44353:46369]=22
    label[46369:48385]=23
    label[48385:50401]=24
    label[50401:]=25
    
    
    data,label=shuffle(imatrix,label, random_state=2)
    train_data=[data,label]
    
    #print(train_data[0].shape)
    #print(train_data[1].shape)

    (x,y) = (train_data[0],train_data[1])
    
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=4)
    x_train=x_train.reshape(x_train.shape[0],200,200,1)
    x_test=x_test.reshape(x_test.shape[0],200,200,1)
    
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')    
    
    x_train/=255
    x_test/=255
    
    #print('x_train_shape:',x_train.shape)
    #print(x_train.shape[0],' train samples')
    #print(x_test.shape[0],' test samples')
    
    y_train=np_utils.to_categorical(y_train,26)
    y_test=np_utils.to_categorical(y_test,26)
    
    #print('labels = ',y_train.shape[1])
    #print(x_train.shape)
    #i=100
    #plt.imshow(x_train[i,0],interpolation='nearest')
    #print('label: ',y_train[i,:])   
    
    model=Sequential()
    model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(200,200,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(200,200,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(200,200,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Dropout(0.10))
    model.add(Dense(26))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',  metrics=['accuracy'])
    
    model.fit(x_train,y_train,batch_size=32,nb_epoch=1, verbose=1, validation_data=(x_test,y_test))
    
    score=model.evaluate(x_test,y_test, verbose=0)
    print('test score: ',score[0])
    print('test accuracy: ',score[1])
    print(model.predict_classes(x_test[1:5]))
    print(y_test[1:5])
    
    model.save('final_model.h5')
            
if __name__=="__main__":
    main()



