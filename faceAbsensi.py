import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'absensi'  #folder for face image
images = []
classNames = []
myList = os.listdir(path)


for cl in myList:
    curlImg = cv2.imread(f'{path}/{cl}')
    images.append(curlImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def faceList(name):
    with open('data.csv', 'r+') as f:  #file csv untuk menyimpan data di excel
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListUnkown = findEncodings(images)
print('encoding complate!')

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListUnkown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListUnkown,encodeFace)
        #print(faceDis)
        matchesIndex = np.argmin(faceDis)
        
        if matches[matchesIndex]:
            name = classNames[matchesIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            faceList(name)

    cv2.imshow('face', img)
    cv2.waitKey(1)

#imgFatih = face_recognition.load_image_file('fatih.jpeg')
#imgFatih = cv2.cvtColor(imgFatih,cv2.COLOR_BGR2RGB)
#imgTest = face_recognition.load_image_file('jokowi.jpeg')
#imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
