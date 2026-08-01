import cv2

file_loc = input("Enter File Location : ")
color=(232,23,54)
thk=4

base_img = cv2.imread(file_loc)
choice = int(input("Enter choice \n 1 -> Draw Rectangle \n 2->  Draw Circle \n 3 -> Draw Line \n 4 -> Draw Text \n Choose number : "))

if choice ==1:
    pt1x = int(input("Enter Point 1 cord x : "))
    pt1y = int(input("Enter Point 1 cord y : "))
    pt2x = int(input("Enter Point 2 cord x : "))
    pt2y = int(input("Enter Point 2 cord y : "))
    rect_img = cv2.rectangle(base_img,(pt1x,pt1y),(pt2x,pt2y),color,thk)
    cv2.imshow("image",rect_img)
    cv2.waitKey(0)
    conf = input("Wanna Save Image ? yes, no : ")
    if conf == "yes":
        f_name = input("Give File Name (without extension) : ")
        f_name=f_name+".png"
        cv2.imwrite(f_name,rect_img)
    else :
        print("File Not Saved ")
    

elif choice == 2:
    pt1 = int(input("Enter Point 1 : "))
    pt2 = int(input("Enter Point 2 : "))
    radius = int(input("Enter Radius : "))
    circle_img = cv2.circle(base_img,(pt1,pt2),radius,color,thk)
    cv2.imshow("image",circle_img)
    cv2.waitKey(0)
    conf = input("Wanna Save Image ? yes, no : ")
    if conf == "yes":
        f_name = input("Give File Name (without extension) : ")
        f_name=f_name+".png"
        cv2.imwrite(f_name,circle_img)
    else :
        print("File Not Saved ")
        
elif choice ==3:
    pt1x = int(input("Enter Point 1 cord x : "))
    pt1y = int(input("Enter Point 1 cord y : "))
    pt2x = int(input("Enter Point 2 cord x : "))
    pt2y = int(input("Enter Point 2 cord y : "))

    line_img = cv2.line(base_img,(pt1x,pt1y),(pt2x,pt2y),color,thk)
    cv2.imshow("image",line_img)
    cv2.waitKey(0)
    conf = input("Wanna Save Image ? yes, no : ")
    if conf == "yes":
        f_name = input("Give File Name (without extension) : ")
        f_name=f_name+".png"
        cv2.imwrite(f_name,line_img)
    else :
        print("File Not Saved ")
        
elif choice == 4:
    pt1 = int(input("Enter Point 1 : "))
    pt2 = int(input("Enter Point 2 : "))
    text=input("Enter Text : ") 
    text_img = cv2.putText(base_img,text,(pt1,pt2),cv2.FONT_HERSHEY_COMPLEX,1.2,color)
    cv2.imshow("image",text_img)
    cv2.waitKey(0)
    conf = input("Wanna Save Image ? yes, no : ")
    if conf == "yes":
        f_name = input("Give File Name (without extension) : ")
        f_name=f_name+".png"
        cv2.imwrite(f_name,text_img)
    else :
        print("File Not Saved ")
else :
    print("Enter correct Choice Among 1,2,3 and 4") 