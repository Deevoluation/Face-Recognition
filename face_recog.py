# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:20:15 2017

@author: Siddhant
"""
import cv2
import numpy as np
import os
import shutil
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto import Random

def face_extractor(img):
    face_classifier = cv2.CascadeClassifier('C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.05,4)
    
    if faces is ():
        return None
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        
    return cropped_face


def encrypt(key, filename):
    chunksize = 64 * 1024
    outputFile =  './encrypted/'+ filename
    filename = './faces/' + filename
    filesize = str(os.path.getsize(filename)).zfill(16)
    IV = Random.new().read(16)

    encryptor = AES.new(key, AES.MODE_CBC, IV)

    with open(filename, 'rb') as infile:
        with open(outputFile, 'wb') as outfile:
            outfile.write(filesize.encode('utf-8'))
            outfile.write(IV)

            while True:
                chunk = infile.read(chunksize)

                if len(chunk) == 0:
                    break
                elif len(chunk) % 16 != 0:
                    chunk += b' ' * (16 - (len(chunk) % 16))

                outfile.write(encryptor.encrypt(chunk))



if __name__ == '__main__':
    #Making the key
    key = str('2486')
    hasher = SHA256.new(key.encode('utf-8'))
    key = hasher.digest()
    
    #Initialise webcam
    cap = cv2.VideoCapture(0)
    count = 0
    try:
        if not os.path.exists('encrypted'):
            os.mkdir('encrypted')
        if not os.path.exists('faces'):
            os.mkdir('faces')
    except OSError:
        print('Error: creating directory data or faces')       
   
    while True:
        ret,frame  = cap.read()
        face = face_extractor(frame)
        if face is not None:
            count += 1
            face_edited = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face_edited = cv2.equalizeHist(face_edited)
            face_resized = cv2.resize(face_edited, (300,300))
            
            #SAVE IMAGE IN DIRECTORY
            file_path = './faces/' + str(count) + '.jpg'
            cv2.imwrite(file_path,face_resized)
            file_path = str(count) + '.jpg'
            encrypt(key, file_path)
            
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face detector',face)
            
        else:
            print("Face not found")
            pass
        if (cv2.waitKey(13) == 13 or count == 300):
            break
           
    cap.release()
    cv2.destroyAllWindows()        
    
    try:
        shutil.rmtree('faces', ignore_errors=True)
    except OSError:
        print('Error: Deleting data')
    
    os.mkdir('faces')
    print ("Samples collected")
