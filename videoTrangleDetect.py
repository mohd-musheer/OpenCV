import cv2


cap = cv2.VideoCapture(0)



while True:
    _,frame=cap.read()
    _,thresh = cv2.threshold(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),120,100,cv2.THRESH_BINARY) 
    
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(frame,contours,-1,(0,255,0),4 )
    
    cv2.imshow("Img",frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    
cap.release()
cv2.destroyAllWindows()