import cv2

img = cv2.imread("my_img.jpeg")

blue = img[:, :, 0]
green = img[:, :, 1]

cv2.imshow("Blue", blue)
cv2.imshow("Green", green)

cv2.waitKey(0)
cv2.destroyAllWindows()