import cv2

img = cv2.imread("my_img.jpeg",0)

cv2.imshow("sjsds",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img)