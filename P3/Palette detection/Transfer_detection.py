import cv2 as cv
import os
import numpy as np

folder_path = "C://Users//jens1//Documents//GitHub//Robotteknologi-3.-semester//P3//Palette detection//Testpics"
dir_list = os.listdir(folder_path)
print(dir_list)

images = []
for file in dir_list:
    image_path = os.path.join(folder_path, file)
    image = cv.imread(image_path)
    images.append(image)

global template
template = cv.imread("C://Users//jens1//Documents//GitHub//Robotteknologi-3.-semester//P3//Palette detection//Checker.png")

def Template(imgpre, imgfur, Template, loc = 0):
    img = imgpre
    if loc != 0:
        print(loc)
        img = imgpre-imgfur
        Template=(imgfur-imgpre)[loc[1]:loc[1] + template.shape[0], 
                                 loc[0]:loc[0] + template.shape[1]]
        cv.imshow("templatein", cv.resize(Template, (1080, 606), interpolation = cv.INTER_LINEAR))
        cv.imshow("imgin", cv.resize(img, (1080, 606), interpolation = cv.INTER_LINEAR))
        cv.waitKey(0)    
    cv.imshow("templateout", cv.resize(Template, (1080, 606), interpolation = cv.INTER_LINEAR))
    cv.imshow("imgout", cv.resize(img, (1080, 606), interpolation = cv.INTER_LINEAR))
    cv.waitKey(0)     
    Temp_match_result = cv.matchTemplate(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.cvtColor(Template, cv.COLOR_BGR2GRAY), cv.TM_CCOEFF_NORMED)
    Location_list = np.where(Temp_match_result >= 0.4)
    Position_list = list(zip(*Location_list[::-1]))
    print(f"PosList: {Position_list}")

    if(len(Position_list)!=0):
        Current_X, Current_Y = Position_list[0]
        Dim=template.shape
        cv.rectangle(img, Position_list[0], (Current_X + Dim[1], Current_Y + Dim[0]), [0,0,255], 25)
        print("FOUND")  
        cv.imshow("template", cv.resize(img, (1080, 606), interpolation = cv.INTER_LINEAR))
        cv.waitKey(0) 
        return Position_list[0]    
    return 

Loc = Template(images[0], 0, template, 0)
Loc = Template(images[0], images[1], template, Loc)
#cv.imshow("Imagestuff1", cv.resize(images[1]-images[0], (1080, 606), interpolation = cv.INTER_LINEAR))
#cv.imshow("Imagestuff2", cv.resize(images[0]-images[1], (1080, 606), interpolation = cv.INTER_LINEAR))

cv.waitKey(0)



#for i in range(len(images)):
#    images[i] = cv.resize(images[i], (1080, 606), interpolation = cv.INTER_LINEAR)
#cv.imshow("Image1", images[0])


#cv.imshow("Image1", images[1]-images[0])
#cv.imshow("Image2", images[2]-images[1])
#cv.imshow("Image3", images[3]-images[2])
#cv.imshow("Image4", images[4]-images[3])
#cv.imshow("Image5", images[5]-images[4])
#cv.imshow("Image6", images[6]-images[5])


