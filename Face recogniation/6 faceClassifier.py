import numpy as np
import cv2

vijay = np.load(r"D:\Code OUTPUTS\Machine Learning Workshop IITDH 20\Face recogniation\Face Datasets\Vijay.npy").reshape((160,50*50*3))
sharukh = np.load(r"D:\Code OUTPUTS\Machine Learning Workshop IITDH 20\Face recogniation\Face Datasets\Sharukh Khan.npy").reshape((160,50*50*3))

data = np.concatenate([vijay,sharukh])
label = np.zeros((320,1))
label[:161] = 0
label[161:] = 1

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(data,label)
person = ['Vijay','Sharukh']

cam = cv2.VideoCapture(0)
fCascade = cv2.CascadeClassifier(r"D:\Code OUTPUTS\Machine Learning Workshop IITDH 20\Face recogniation\Face Datasets\haarcascade_frontalface_default.xml")
while True:
    ret, image = cam.read()  
    imGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = fCascade.detectMultiScale(imGray,1.3,5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        face = image[y:y+h,x:x+w,:]
        resizedFace = cv2.resize(face,(50,50))
        predict = classifier.predict([resizedFace.flatten()])
        output = person[int(predict)]
        cv2.putText(image,output,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Face Classification',image)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()

