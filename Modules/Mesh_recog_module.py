import mediapipe as mp
import cv2
import time

class MeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, refine=True, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine = refine
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpMesh = mp.solutions.face_mesh
        self.mesh = self.mpMesh.FaceMesh(self.staticMode, self.maxFaces, refine_landmarks=refine, min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon)
        self.ballSpec = self.mpDraw.DrawingSpec(color=(255, 186, 226), thickness=1, circle_radius=1)
        self.connectSpec = self.mpDraw.DrawingSpec(color=(157, 78, 13), thickness=1, circle_radius=1)

    def findMesh(self, frame, draw=True):
        # frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.mesh.process(frame)

        if self.results.multi_face_landmarks:
            faces = []
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, faceLms, self.mpMesh.FACEMESH_CONTOURS, landmark_drawing_spec=self.ballSpec,connection_drawing_spec=self.connectSpec)
                face = []

                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(frame, f"{id}", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
                    face.append([x, y])
                faces.append(face)
        return frame, faces

    def getPos(self, frame, draw=True):
        lmList = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    lmList.append([id, x, y])
                    # if draw:
                        # cv2.circle(frame, (x, y), 1, (0, 0, 255), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    detector = MeshDetector()

    ptime = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame, faces = detector.findMesh(frame)
        if len(faces) != 0:
            print(faces[0])

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (36, 178, 255), 3)
        cv2.imshow("Face Mesh", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()