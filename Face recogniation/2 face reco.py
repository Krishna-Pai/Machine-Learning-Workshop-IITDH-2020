import cv2

image = cv2.imread('Threemen.jpg')                         #reading image
fCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #getting grayscale image
grayImage = cv2.equalizeHist(grayImage)
faces = fCascade.detectMultiScale(grayImage,1.6,4)     #find faces, change values for acc, KNN
print(faces)

face1 = image[235:235+165,478:478+165,1]                                       # extratcing red plane
cv2.imshow('Sample image1',face1)                        #view image
face2 = image[142:142+165,248:248+165,1]                                       # extratcing red plane
cv2.imshow('Sample image2',face2)                        #view image
face3 = image[141:141+165,760:760+165,1]                                       # extratcing red plane
cv2.imshow('Sample image3',face3)                        #view image
cv2.imshow('Sample image',image)                        #view image
cv2.waitKey(0)                                          #press ENTER to stop
cv2.destroyAllWindows()

