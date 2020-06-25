import numpy as np
import cv2

cam = cv2.VideoCapture(0)
name = input('Enter your name:')

fCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceData = []
captureCount = 0
faceCount = 0

while True:
    ret, image = cam.read()  
    imGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = fCascade.detectMultiScale(imGray,1.3,5)
    
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image,(x,y), (x+w, x+h), (0,0,255),5)
        face = image[y:y+h,x:x+w,:]
        resizedFace = cv2.resize(face,(50,50))
        
        captureCount +=1
        if captureCount/20 == 1:
            faceData.append(resizedFace)
            captureCount = 0
            faceCount += 1
        cv2.putText(image, 'Captured {} faces'.format(faceCount), (10,10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    if cv2.waitKey(1) == 27 or faceCount > 20:
        cam.release()
        cv2.destroyAllWindows()
        break
    cv2.imshow('Capturing Face Data', image)

data = np.array(faceData)
np.save(name,data)