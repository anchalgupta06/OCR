# OCR
OCR for handwriting recognition
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:05:03 2018

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
from keras.models import load_model

def main():
    im=cv2.imread('C:\\Users\\Anchal\\Downloads\\t1.jpeg')
    im=cv2.resize(im,(200,200))
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
    
    #cv2.imshow('image',im2)
    cv2.imshow('image1',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    ima=array(thresh)
    m,n=ima.shape[0:2]
    print(m)
    print(n)
    print(ima)
    ima=ima.reshape(1,200,200,1)
    ima=ima.astype('float32')
    ima/=255
    print(ima)
    model=load_model('final_model.h5')
    pre=model.predict_classes(ima)
    print (f(int(pre)))
    

def f(x):
    return {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z',
    }[x]
    
if __name__=="__main__":
    main()
