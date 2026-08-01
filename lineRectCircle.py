import cv2

img = cv2.imread("my_img.jpeg")
print(img.shape[:2])
color=(241,145,21)
pt1=(40,10)
pt2=(150,150)

# cv2.line(img,pt1,pt2,color,4)
cv2.rectangle(img,pt1,pt2,color,2)
cv2.circle(img,(150,150),50,color,2)
cv2.circle(img,(50,150),50,color,2)


cv2.imshow("sjsds",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
