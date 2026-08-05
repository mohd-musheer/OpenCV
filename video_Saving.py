import cv2

camera = cv2.VideoCapture(0)

frame_w=int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h=int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

codec = cv2.VideoWriter.fourcc(*'XVID')
recorder = cv2.VideoWriter("MyVideo.avi",codec,30,(frame_w,frame_h))
while True:
    success,image=camera.read()
    recorder.write(image)
    cv2.imshow("Live : ",image)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
camera.release()
recorder.release()
cv2.destroyAllWindows()