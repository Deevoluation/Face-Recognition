# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 01:43:24 2017

@author: Siddhant
"""

import cv2
import numpy as np

# We point OpenCV's CascadeClassifier function to where our
# classifier (XML file format is stored)
face_classifier = cv2.CascadeClassifier('C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')

# Load our image then convert to grayscale
image = cv2.imread('modi_face.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a tuple
# It stores teh top left coordinate and the bottom left
faces = face_classifier.detectMultiScale(gray,1.3,2)
#cv2.imshow('Face detctor',image)
#cv2.waitKey(0)
# When no faces detected ,faceClassifier returns an empty tuple
if faces is ():
    print('no faces found')

# We iterate threough our faces array and draw a rectangel
# over each face in faces
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Face detctor',image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()