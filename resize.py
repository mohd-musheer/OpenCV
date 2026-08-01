import cv2

img = cv2.imread("my_img.jpeg")

resized_img = cv2.resize(img,(400,400))
cv2.imshow("Green", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()