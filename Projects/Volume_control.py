import mediapipe as mp
import cv2
import time
import numpy as np
import osascript
from Modules import Hand_recog_module as hrm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0

detector = hrm.handDetector(detectionConf=0.7)



while cap.isOpened():
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPos(frame, draw=False)
    relativeLength = 0
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.line(frame, (x1, y1), (x2, y2), (32, 215, 255), 3, cv2.FILLED)
        cv2.circle(frame, (x1, y1), 15, (215, 215, 32), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (x2, y2), 15, (215, 215, 32), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 10, (215, 215, 32), cv2.FILLED)

        length = np.hypot(x2 - x1, y2 - y1)
        relativeLength = int(np.interp(length, [50, 250], [0, 100]))
        osascript.run(f'set volume output volume {relativeLength}')
        if length < 50:
            osascript.run('set volume output volume 0')
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)


    cv2.rectangle(frame, (50, int(400 - relativeLength * 2.5)), (85, 400), (215, 215, 32), cv2.FILLED)
    cv2.rectangle(frame, (50, 150), (85, 400), (32, 215, 255), 3)
    cv2.putText(frame, f'Volume %:{int(relativeLength)}', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (215, 215, 32), 2)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (180, 23, 245), 2)
    cv2.imshow('Volume Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
