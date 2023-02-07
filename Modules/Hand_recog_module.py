import cv2
import numpy as np
import time
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionConf
        self.trackCon = trackConf

        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPos(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = []

        if self.results.multi_hand_landmarks:
            xList = []
            yList = []
            bbox = []

            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # height width channels
                h, w, c = img.shape
                # position of the center
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                # draw circle at fingertips
                if id % 4 == 0 and id != 0 and draw:
                    cv2.circle(img, (cx, cy), 15, (23, 226, 80), cv2.FILLED)

                xList.append(cx)
                yList.append(cy)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDis(self, p1, p2, frame, thres, draw=False):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = np.hypot(x2 - x1, y2 - y1)

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (240, 140, 43), 3)
            cv2.circle(frame, (x1, y1), 15, (43, 240, 201), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (43, 240, 201), cv2.FILLED)
            cv2.circle(frame, (cx, cy), 15, (200, 240, 12), cv2.FILLED)

            if length < thres:
                cv2.circle(frame, (cx, cy), 15, (240, 43, 43), cv2.FILLED)

        return length, frame, [x1, y1, x2, y2, cx, cy]

def main():
    # Dummy Code for hand tracking projects
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = detector.findHands(img=frame)
        lmList, bbox = detector.findPos(img=frame)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (180, 23, 245), 3)
        cv2.imshow('Hand', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
