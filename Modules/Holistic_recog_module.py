import mediapipe as mp
import cv2
import time

# Holistic
mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic()
mpDraw = mp.solutions.drawing_utils

class HolisticDetector:
    def __int__(self, mode=False, segment=True, refine=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.segment = segment
        self.refine = refine
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(self.mode, enable_segmentation=self.segment, refine_face_landmarks=self.refine, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHolistic(self, frame, draw=True):
        # frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame)

        ballSpec = mpDraw.DrawingSpec(color=(255, 186, 226), thickness=1, circle_radius=1)
        connectSpec = mpDraw.DrawingSpec(color=(157, 78, 13), thickness=1, circle_radius=1)

        poseLand = mpDraw.DrawingSpec(color=(129, 255, 38), thickness=2, circle_radius=2)
        poseConnect = mpDraw.DrawingSpec(color=(38, 255, 197), thickness=3, circle_radius=1)

        if results.pose_landmarks:
            if draw:
                mpDraw.draw_landmarks(frame, results.face_landmarks, mpHolistic.FACEMESH_CONTOURS, landmark_drawing_spec=ballSpec, connection_drawing_spec=connectSpec)
                mpDraw.draw_landmarks(frame, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS, landmark_drawing_spec=poseLand, connection_drawing_spec=poseConnect)
                mpDraw.draw_landmarks(frame, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(frame, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS)

        return frame


def main():
    cap = cv2.VideoCapture(0)
    detector = HolisticDetector()

    ptime = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = detector.findHolistic(frame)

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (36, 178, 255), 3)
        cv2.imshow("Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()