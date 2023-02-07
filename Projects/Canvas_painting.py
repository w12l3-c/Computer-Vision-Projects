import cv2
import time
import numpy as np
import os
from Modules import Hand_recog_module as hrm

folderPath = 'paints'
imageList = sorted(os.listdir(folderPath))
overlayList = []
for image in imageList:
    image = cv2.imread(f'{folderPath}/{image}')
    image = cv2.resize(image, (int(image.shape[1]), int(image.shape[0]*0.85)), interpolation=cv2.INTER_AREA)
    overlayList.append(image)

wCam, hCam = 1280, 760
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

detector = hrm.handDetector(detectionConf=0.7)
mode = 0
color = (217, 213, 208)   # grey
brushThickness = 15
eraserThickness = 50
xp, yp = 0, 0
canvas = np.zeros((hCam-40, wCam, 3), np.uint8)

ptime = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    frame[0:overlayList[0].shape[0], 0:overlayList[0].shape[1]] = overlayList[mode]
    lmList, bbox = detector.findPos(frame, draw=False)

    if len(lmList) != 0:
        # index finger
        x1, y1 = lmList[8][1:]
        # middle finger
        x2, y2 = lmList[12][1:]

        # Detecting Finger
        fingers = detector.fingersUp()

        # Selecting with index and middle
        # cv2.putText(frame, f'{x1}, {y1}', (x1+20, y1), cv2.FONT_HERSHEY_PLAIN, 1, (180, 23, 245), 1)
        if fingers[1] and fingers[2]:
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), color, cv2.FILLED)

            # when finger is on the top banner
            if 20 < y1 < 110:
                # home
                if 100 < x1 < 190:
                    mode = 0
                    frame[0:overlayList[mode].shape[0], 0:overlayList[mode].shape[1]] = overlayList[mode]
                    color = (217, 213, 208)  # grey
                    canvas = np.zeros((hCam-40, wCam, 3), np.uint8)
                # red colour
                elif 440 < x1 < 560:
                    mode = 1
                    frame[0:overlayList[mode].shape[0], 0:overlayList[mode].shape[1]] = overlayList[mode]
                    color = (17, 47, 242)  # red
                # green colour
                elif 660 < x1 < 770:
                    mode = 2
                    frame[0:overlayList[mode].shape[0], 0:overlayList[mode].shape[1]] = overlayList[mode]
                    color = (88, 242, 61)  # green
                # blue colour
                elif 880 < x1 < 990:
                    mode = 3
                    frame[0:overlayList[mode].shape[0], 0:overlayList[mode].shape[1]] = overlayList[mode]
                    color = (245, 148, 44)  # blue
                # eraser
                elif 1110 < x1 < 1210:
                    mode = 4
                    frame[0:overlayList[mode].shape[0], 0:overlayList[mode].shape[1]] = overlayList[mode]
                    color = (102, 66, 72)   # purple

        # Drawing with index finger only
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (x1, y1), 15, color, cv2.FILLED)

            # to prevent drawing from the top banner
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            length = np.sqrt((x1 - xp) ** 2 + (y1 - yp) ** 2)
            if color == (102, 66, 72):
                cv2.line(frame, (xp, yp), (x1, y1), (102, 66, 72), eraserThickness)
                cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
            else:
                if length < 75:
                    cv2.line(frame, (xp, yp), (x1, y1), color, brushThickness)
                    cv2.line(canvas, (xp, yp), (x1, y1), color, brushThickness)

            xp, yp = x1, y1

    # Convert canvas to black and white
    frameGrey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Invert the image with threshold
    _, frameInv = cv2.threshold(frameGrey, 50, 255, cv2.THRESH_BINARY_INV)
    # Convert the image to BGR
    frameInv = cv2.cvtColor(frameInv, cv2.COLOR_GRAY2BGR)
    # Bitwise the frame
    frame = cv2.bitwise_and(frame, frameInv)    # and will add the black part of the inverted image
    frame = cv2.bitwise_or(frame, canvas)   # or will select the coloured part in the canvas and display over the black

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    # cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (180, 23, 245), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
