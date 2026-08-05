import cv2 

cap = cv2.VideoCapture(0)

while True:
    ret,img=cap.read()
    ret,t_fr = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),50,200,cv2.THRESH_BINARY)
    cv2.imshow("Video",t_fr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()