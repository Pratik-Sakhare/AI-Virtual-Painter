import cv2
import mediapipe
import time
import numpy as np
import os
import HandTrackingModule as htm

#################################
brusht = 15
rubbert = 100
#################################

folderPath = "VirtualPainter"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
header = overlayList[-1]
drawColor = (0, 0, 0)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    #1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:
        #print(lmList[8], lmList[12])

        #tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]



        #3. Check which finger is up
        fingers = detector.fingerUp()
        #print(fingers)

        #4. If selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("Selection Mode")
            #checking for click
            if y1 < 133:
                if 200 < x1 < 330:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 450 < x1 < 600:
                    header = overlayList[1]
                    drawColor = (0, 123, 0)
                elif 700 < x1 < 850:
                    header = overlayList[2]
                    drawColor = (157, 0, 0)
                elif 950 < x1 < 1280:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.line(img, (x1, y1), (x2, y2), drawColor, 3)



        #5. If Drawing mode - index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, -1)
            #print("Drawing Mode")
            if xp ==0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, rubbert)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, rubbert)
            else:

                cv2.line(img, (xp, yp), (x1, y1), drawColor, brusht)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brusht)

            xp, yp = x1, y1


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)




    #setting head image
    img[0:133, 100:1180] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("AI VIRTUAL PAINTER", img)
    cv2.imshow("DRAWING BOARD", imgCanvas)

    k = cv2.waitKey(1)
    if cv2.waitKey(1) and k == 27:
        break

cap.release()
cv2.destroyAllWindows()