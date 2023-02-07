import cv2
import numpy as np
import time
import mediapipe as mp

# Palm Detection
# Hand landmarking (21 points)

mp_hands = mp.solutions.hands
# until tracking confidence is low, it will detect again
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    image_RBG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_RBG)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                # height width channels
                h, w, c = frame.shape
                # position of the center
                cx, cy = int(lm.x*w), int(lm.y*h)
                # draw circle at fingertips
                if id % 4 == 0 and id != 0:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (180, 23, 245), 3)
    cv2.imshow('Hand', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


