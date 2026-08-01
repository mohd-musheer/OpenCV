import cv2

img = cv2.imread("my_img.jpeg")

cv2.putText(img,"Musheer",(40,200),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1.2,(124,0,50))
cv2.imshow("sjsds",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
