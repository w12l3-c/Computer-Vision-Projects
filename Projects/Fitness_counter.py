import mediapipe as mp
import cv2
import time

import numpy as np

from Modules import Pose_recog_module as prm

wCam, hCam = 1280, 760

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = prm.PoseDetector()
count = 0
direction = 0   # 0 for up, 1 for down, need up and down to count as 1

ptime = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = detector.findPose(frame, draw=False)
    lmList = detector.getPos(frame, draw=False)
    if len(lmList) != 0:
        # left arm is odd 11 13 15
        laAngle = detector.findAngle(frame, 11, 13, 15)
        # right arm is even 12 14 16
        raAngle = detector.findAngle(frame, 12, 14, 16)
        # left leg is odd 23 25 27
        llAngle = detector.findAngle(frame, 23, 25, 27)
        # right leg is even 24 26 28
        rlAngle = detector.findAngle(frame, 24, 26, 28)

        per = np.interp(laAngle, (50, 160), (0, 100))
        bar = np.interp(laAngle, (50, 160), (650, 100))

        # check the angle percentage
        if per == 100:
            color = (0, 255, 0)
            if direction == 0:
                count += 0.5
                direction = 1
        if per == 0:
            color = (200, 180, 0)
            if direction == 1:
                count += 0.5
                direction = 0

        cv2.rectangle(frame, (1100, int(bar)), (1125, 650), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (1100, 100), (1125, 650), (156, 180, 0), 3)
        cv2.putText(frame, f'{int(per)}%', (1130, int(bar)+10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 190, 0), 2)
        cv2.rectangle(frame, (20, hCam-140), (250, hCam-90), (255, 23, 0), cv2.FILLED)
        cv2.putText(frame, f'Count: {int(count)}', (25, hCam-100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 4)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (180, 23, 245), 2)
    cv2.imshow('Fitness', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()