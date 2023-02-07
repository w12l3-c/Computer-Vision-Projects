import mediapipe as mp
import cv2
import time
import images
import os
from Modules import Hand_recog_module as hrm

wCam, hCam = 1280, 760

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# saving the image paths in a list
folderPath = 'hands_image'
imageList = sorted(os.listdir(folderPath))
rp = 0.20

overlayList = []
for image in imageList:
    image = cv2.imread(f'{folderPath}/{image}')
    image = cv2.resize(image, (int(image.shape[1]*rp), int(image.shape[0]*rp)), interpolation=cv2.INTER_AREA)
    overlayList.append(image)

# defining the detector from module
detector = hrm.handDetector(detectionConf=0.7)

ptime = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPos(frame, draw=False)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(8, 24, 4):
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        h, w, c = overlayList[totalFingers - 1].shape
        frame[0:h, 0:w] = overlayList[totalFingers - 1]
        cv2.putText(frame, f'Fingers:{totalFingers}', (10, hCam-50), cv2.FONT_HERSHEY_PLAIN, 2, (180, 23, 245), 2)
        cv2.rectangle(frame, (wCam-40, 20), (wCam - 20, 40), (0, 255, 255), cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (180, 23, 245), 2)
    cv2.imshow('Finger Counting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


