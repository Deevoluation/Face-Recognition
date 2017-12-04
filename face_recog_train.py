# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:57:54 2017

@author: Siddhant
"""
import cv2
import numpy as np
import os
import shutil
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto import Random
from os import listdir
from os.path import isfile, join


def decrypt(key, filename):
    chunksize = 64 * 1024
    outputFile = './decrypted/' + filename
    filename = './encrypted/' + filename
    with open(filename, 'rb') as infile:
        filesize = int(infile.read(16))
        IV = infile.read(16)

        decryptor = AES.new(key, AES.MODE_CBC, IV)

        with open(outputFile, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)

                if len(chunk) == 0:
                    break

                outfile.write(decryptor.decrypt(chunk))
            outfile.truncate(filesize)
    return outputFile
if __name__ == '__main__':
    
    try:
        if not os.path.exists('decrypted'):
            os.mkdir('decrypted')
        
    except OSError:
        print('Error: creating directory decrypted')  
    
    while(True):
        key = str(input('Enter the decryption key to gain access : '))
        if(key == '2486'):
            break
    
    hasher = SHA256.new(key.encode('utf-8'))
    key = hasher.digest()
   
    data_path = './encrypted/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) ]
    
    Training_Data, Labels = [], []
    
    #numpy array for training data
    
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        #decrypt the images
        filename = onlyfiles[i]
        output_path = decrypt(key, filename)
        images = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype = np.uint8))
        Labels.append(onlyfiles[i].split('.')[0])
        
    Labels = np.asarray(Labels, dtype = np.int32)
    
    #Initialize and Train model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print ('Model trained successfully')
    '''
    try:
        shutil.rmtree('decrypted', ignore_errors=True)
    except OSError:
        print('Error: Deleting data')
    '''    
    
    
    def face_detector(img,size = 0.5):
        face_classifier = cv2.CascadeClassifier('C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.05, 4)
        if faces is ():
            return img, []
        
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi,(300,300))
        return img,roi
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        image,face = face_detector(frame)
        
        try:
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            # result comprises of a tuple of label and confidence value
            results = model.predict(face)
            confidence = 0
            if results[1] < 300:
                confidence = int(100 * (1-(results[1])/300))
            cv2.putText(image, str(confidence)+' % confident',(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,120,150),2)
            if confidence > 78:
                
                cv2.putText(image,"Unlocked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face Cropper', image)
            else:
                cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) 
                cv2.imshow('Face Cropper',image)
        
        except:
            cv2.putText(image,"No Face Found",(220,120),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('Face Cropper', image)
            pass
        
        if cv2.waitKey(1) == 13:
            break
            
    cap.release()
    cv2.destroyAllWindows()
        