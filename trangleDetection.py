import cv2
# import numpy as np
# arr1 = np.zeros((300,300),dtype='uint8')
# arr1[:,:]=255
# cv2.rectangle(arr1,(100,100),(200,200),(0,0,0),3)

img = cv2.imread("trangle.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
counter,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,counter,-1,(0,0,255),4)
cv2.imshow("img",img)

cv2.waitKey(0)
cv2.destroyAllWindows()