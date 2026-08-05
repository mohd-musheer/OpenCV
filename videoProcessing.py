import cv2

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret:
        print("Problem with Camera")
        
    img = cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)
    # blurred = cv2.GaussianBlur(frame,(21,21),19)
    # cv2.imshow("Video",frame)

    cv2.imshow("Video",img)
    # cv2.imshow("Video",blurred)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        print("Quitting.....")
        break
    
cap.release()
cv2.destroyAllWindows()        