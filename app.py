import cv2
import numpy as np
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

myPoints = []

def findColor(img):
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    newPoints = []
    lower = np.array([100, 21, 37])
    upper = np.array([148, 249, 160])
    mask = cv2.inRange(imgHsv, lower, upper)
    x,y=getContours(mask)
    cv2.circle(imgResult,(x,y),10,(255,255,255),cv2.FILLED)
    if x!=0 and y!=0:
        newPoints.append([x,y])
    #cv2.imshow("Image",mask)
    return newPoints

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2,y

def drawOnCanvas(myPoints):
    for point in myPoints:
        cv2.circle(imgResult,(point[0],point[1]),10,(255,0,0),cv2.FILLED)


while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    imgResult = img.copy()
    newPoints = findColor(img)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        drawOnCanvas(myPoints)
    cv2.imshow("Result",imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break