import cv2

img = cv2.imread("my_img.jpeg")

img = cv2.flip(img,1)
cv2.imshow("sjsds",img)
cv2.waitKey(0)
cv2.destroyAllWindows()