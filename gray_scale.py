import cv2


cap = cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    edge = cv2.Canny(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),50,150)
    
    cv2.imshow("video",edge)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()

cv2.destroyAllWindows()