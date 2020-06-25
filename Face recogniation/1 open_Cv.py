import cv2

image = cv2.imread('Threemen.jpg')                         #reading image                     
print(image.shape)                                         #resolution and no. of layers
imageR = image[:,:,0]                                       # extratcing red plane

cv2.imshow('Sample image',imageR)                        #view image
cv2.waitKey(0)                                          #press ENTER to stop
cv2.destroyAllWindows()

