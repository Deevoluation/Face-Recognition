# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:20:15 2017

@author: Siddhant
"""
import cv2
import numpy as np


def face_extractor(img):
    face_classifier = cv2.CascadeClassifier('C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.05,4)
    
    if faces is ():
        return None
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        
    return cropped_face

#Initialise webcam
cap = cv2.VideoCapture(0)
count = 0
#x = input()
#Collect 100 samples of user's face from webcam
while True:
    
    ret,frame  = cap.read()
    face = face_extractor(frame)
    if face is not None:
        count += 1
        face_edited = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face_edited = cv2.equalizeHist(face_edited)
        face_resized = cv2.resize(face_edited, (200,200))
    
        #SAVE IMAGE IN DIRECTORY
        file_path = 'C:\\Users\\Siddhant\\jupyter_notebook\\face_detection\\faces\\' + str(count) + '.jpg'
        cv2.imwrite(file_path,face_resized)
        
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face detector',face)
        
    else:
        print("Face not found")
        pass
    if (cv2.waitKey(13) == 13 or count == 200):
        break
        
cap.release()
cv2.destroyAllWindows()
print ("Samples collected")
