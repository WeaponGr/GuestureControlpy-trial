import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

#=========================================================================================

app = wx.App(False)
(sx, sy) = wx.GetDisplaySize()
(camx , camy) = (320, 240) 

mouse = Controller()
lowerBound =(120, 20, 20)
upperBound =(220, 50, 50)
kernalOpen = np.ones((3, 3))
kernalClose = np.ones((5, 5))
#font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
cam.set(3, camx)
cam.set(4, camy)

mLocOld = np.array([0, 0])
mouseLoc = np.array([0, 0])
DampingFactor = 2

#mouseLoc = mLocOld + (targetloc - mLocOld)//DampingFactor

#=========================================================================================

while True:
    ret, img = cam.read()
    #img = cv2.resize(img,(600, 300))

    #convert Bgr to HSV
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #create Mask
    mask = cv2.inRange(imgRBG, lowerBound, upperBound)
    #Morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernalOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE,kernalClose)
    
    maskFinal = maskOpen

    conts = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    #cv2.drawContours(img, conts, -1, (255, 0, 0), 3)

    #=========================================================================================

    if (len(conts) == 2):
        mouse.release(Button.left)
        x1, y1, w1, h1 = cv2.boundingRect(conts[0])
        x2, y2, w2, h2 = cv2.boundingRect(conts[1])
        cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
        cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
        cx1 = x1+w1//2
        cy1 = y1+h1//2
        cx2 = x2+w2//2
        cy2 = y2+h2//2
        cx = (cx1+cx2)//2
        cy = (cy1+cy2)//2
        cv2.line(img, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
        mouseLoc = mLocOld + ((cx, cy) - mLocOld)//DampingFactor
        mouse.position = (sx-(mouseLoc[0]*sx//camx), mouseLoc[1]*sy//camy)

        while mouse.position != (sx-(mouseLoc[0]*sx//camx), mouseLoc[1]*sy//camy):
            pass
        mLocOld = mouseLoc

    #=========================================================================================

    elif(len(conts)==1):
        x, y, w, h = cv2.boundingRect(conts[0])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0),2)
        cx = x+w//2
        cy = y+h//2
        cv2.circle(img, (cx, cy), (w+h)//4, (0, 0, 255),2)
        mouseLoc = mLocOld + ((cx, cy) - mLocOld)//DampingFactor
        mouse.position = (sx-(mouseLoc[0]*sx//camx), mouseLoc[1]*sy//camy) 
        mouse.press(Button.left)
        while mouse.position != (sx-(mouseLoc[0]*sx//camx), mouseLoc[1]*sy//camy):
            pass
        mLocOld = mouseLoc

    #=========================================================================================
   
    #for i in range(len(conts)):
        #x, y, w, h = cv2.boundingRect(conts[i])
        #cv2.rectangle(img, (x, y),(x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("maskCLOSE", maskClose)
    #cv2.imshow("maskOpen", maskOpen)
    #cv2.imshow("mask", mask)
    cv2.imshow("cam", img)

    #=========================================================================================

    if cv2.waitKey(20) & 0xff == ord('q'):
        break

    #=========================================================================================

cv2.destroyAllWindows()

#=========================================================================================