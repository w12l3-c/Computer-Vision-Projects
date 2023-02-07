import cv2
import mediapipe as mp
import time
import math

# Pose Detection
mpPose = mp.solutions.pose
pose = mpPose.Pose()    # kind of the same param with Hands except with upper body only and smooth landmarkers
mpDraw = mp.solutions.drawing_utils

class PoseDetector():
    def __init__(self, mode=False, upperBody=False, smooth=True, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, 1, self.smooth, False, self.smooth, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, draw=True):
        #frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frame)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return frame

    def getPos(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 200), cv2.FILLED)
        return self.lmList

    def findAngle(self, frame, p1, p2, p3, draw=True):
        # Get landmarks
        _, x1, y1 = self.lmList[p1][:]
        _, x2, y2 = self.lmList[p2][:]
        _, x3, y3 = self.lmList[p3][:]

        # Calculate Angle
        if p1%2 != 0 and p2%2 != 0 and p3%2 != 0:
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        else:
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2))
        if angle < 0:
            angle += 360
        if angle > 360:
            angle -= 360

        # Draw
        if draw:
            color = (255, 45, 68)
            color2 = (28, 255, 62)
            cv2.line(frame, (x1, y1), (x2, y2), color2, 3)
            cv2.line(frame, (x3, y3), (x2, y2), color2, 3)
            cv2.circle(frame, (x1, y1), 10, color, cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, color, cv2.FILLED)
            cv2.circle(frame, (x3, y3), 10, color, cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, color)
            cv2.circle(frame, (x2, y2), 15, color)
            cv2.circle(frame, (x3, y3), 15, color)
            cv2.circle(frame, (x1, y1), 18, color)
            cv2.circle(frame, (x2, y2), 18, color)
            cv2.circle(frame, (x3, y3), 18, color)
            cv2.putText(frame, f"{int(angle)}", (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        return angle



def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    ptime = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame = detector.findPose(frame)
        lmList = detector.getPos(frame)
        if len(lmList) != 0:
            print(lmList)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (180, 23, 245), 3)
        cv2.imshow('Pose', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()