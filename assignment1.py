import cv2
try :
    loc = input("Enter File Location : ")
    img = cv2.imread(loc)

except Exception as e:
    print("Enter valid loc : ",e)
    
a = int(input(" Enter operation to perform \n 1 : gray Scale \n 2 : show\n Choose : "))  

if a ==1:

    gray_scale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    b = input("Gray Scale Done\n wanna see image ? yes, no : ")
    if b == "yes":
        cv2.imshow("base image",gray_scale_img)
        cv2.waitKey(0)
    else:
        pass
            
else:
        cv2.imshow("base image",img)
        cv2.waitKey(0)
        
        
c = input("Wanna Save Image ? yes, no : ")

if c =="yes":
    file_name = input("Enter file name to save image : ")
    cv2.imwrite(file_name,gray_scale_img)
    print("File Saved Successfully")
else:
    print("File not saved")