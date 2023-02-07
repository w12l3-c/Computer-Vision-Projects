import cv2
import mediapipe as mp
import time
import math

# Pose Detection
mpPose = mp.solutions.pose
pose = mpPose.Pose()    # kind of the same param with Hands except with upper body only and smooth landmarkers
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ptime = 0

while cap.isOpened():
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    # print(results.landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id, cx, cy, lm)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 200), cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (180, 23, 245), 3)
    cv2.imshow('Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyWindow('Pose')