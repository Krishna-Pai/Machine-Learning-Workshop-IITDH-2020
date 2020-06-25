import cv2

image = cv2.imread('Threemen.jpg')                         #reading image
fCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #getting grayscale image
grayImage = cv2.equalizeHist(grayImage)
faces = fCascade.detectMultiScale(grayImage,1.6,4)     #find faces, change values for acc, KNN

for(x,y,w,h) in faces:
    image= cv2.rectangle(image,(x,y),(x+w,y+h), (0,0,255),2)
    
cv2.imshow('Sample image',image)                        #view image
cv2.waitKey(0)                                          #press ENTER to stop
cv2.destroyAllWindows()