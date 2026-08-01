import cv2

img = cv2.imread("my_img.jpeg",0)

(h,w)=img.shape[:2]
center = (w//2,h//2)

rotated_img= cv2.warpAffine(img, cv2.getRotationMatrix2D(center,90,1.0) ,(w,h))

cv2.imshow("sjsds",rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
