import cv2

img = cv2.imread("my_img.jpeg",0)


start_y=100
end_y=400
start_x=100
end_x = 400
cropped= img[start_y:end_y,start_x:end_y]

cv2.imshow("sjsds",cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
