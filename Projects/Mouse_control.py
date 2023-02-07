import cv2
import time
import numpy as np
import pyautogui
from Modules import Hand_recog_module as hrm


detector = hrm.handDetector(detectionConf=0.7, maxHands=1)

wCam, hCam = 1280, 760
wScr, hScr = pyautogui.size()
frameR = 150
smoothening = 7

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

ptime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Find hand landmarks
    frame = detector.findHands(frame)
    lmList, bbox = detector.findPos(frame, draw=False)

    if len(lmList) != 0:
        # index finger
        x1, y1 = lmList[8][1:]
        # middle finger
        x2, y2 = lmList[12][1:]
        # ring finger
        x3, y3 = lmList[16][1:]
        # pinky finger
        x4, y4 = lmList[20][1:]

        # Check fingers up
        fingers = detector.fingersUp()

        # Create range
        cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 180, 45), 2)

        # Index finger mouse move (moving mode)
        if fingers[1] and fingers[2] == False:
            # Convert coordinates
            xRel = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            yRel = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # Smoothen the values
            cLocX = pLocX + (xRel - pLocX) / smoothening
            cLocY = pLocY + (yRel - pLocY) / smoothening

            # Move mouse
            pyautogui.moveTo(cLocX, cLocY)
            cv2.circle(frame, (x1, y1), 10, (180, 200, 12), cv2.FILLED)

            # Update previous location
            pLocX, pLocY = cLocX, cLocY

        # Both fingers mouse click (clicking mode)
        if fingers[1] and fingers[2]:
            length, frame, info = detector.findDis(8, 12, frame, 50, draw=True)
            pyautogui.click()

        # Three fingers scroll
        if fingers[1] and fingers[2] and fingers[3] and fingers[4] == fingers[0] == False:
            cv2.circle(frame, (lmList[8][0], lmList[8][1]), 15, (200, 240, 12), cv2.FILLED)
            cv2.circle(frame, (lmList[12][0], lmList[12][1]), 15, (200, 240, 12), cv2.FILLED)
            cv2.circle(frame, (lmList[16][0], lmList[16][1]), 15, (200, 240, 12), cv2.FILLED)
            pyautogui.scroll(20)

        # Four fingers drag
        # if fingers[1] and fingers[2] and fingers[3] and fingers[4] and fingers[0] == False:
        #     cv2.circle(frame, (lmList[8][0], lmList[8][1]), 15, (90, 240, 12), cv2.FILLED)
        #     cv2.circle(frame, (lmList[12][0], lmList[12][1]), 15, (90, 240, 12), cv2.FILLED)
        #     cv2.circle(frame, (lmList[16][0], lmList[16][1]), 15, (90, 240, 12), cv2.FILLED)
        #     cv2.circle(frame, (lmList[20][0], lmList[20][1]), 15, (90, 240, 12), cv2.FILLED)
        #     xMid, yMid = (x2 + x3) // 2, (y2 + y3) // 2
        #     xRel = np.interp(xMid, (frameR, wCam - frameR), (0, wScr))
        #     yRel = np.interp(yMid, (frameR, hCam - frameR), (0, hScr))
        #
        #     cLocX = pLocX + (xRel - pLocX) / smoothening
        #     cLocY = pLocY + (yRel - pLocY) / smoothening
        #
        #     pyautogui.dragTo(cLocX, cLocY)
        #     pLocX, pLocY = cLocX, cLocY

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 2, cv2.FONT_HERSHEY_PLAIN, (64, 240, 32), 2)
    cv2.imshow("Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



