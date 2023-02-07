import cv2
import mediapipe as mp
import time


# Face Detection
class FaceDetector():
    def __init__ (self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, frame, draw=True):
        #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(frame)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(f"{id},{detection}: {detection.score}")
                # print(detection.location_data.relative_bounding_box)
                bound = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bound.xmin * iw), int(bound.ymin * ih), int(bound.width * iw), int(bound.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.mpDraw.draw_detection(frame, detection)
                    frame = self.fancyDraw(frame, bbox)
                    cv2.putText(frame, f'Human {id+1}: {int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (97, 56, 232), 2)

        return frame, bboxs

    def fancyDraw(self, frame, bbox, l=75, t=15, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        colour = (0, 255, 255)
        cv2.rectangle(frame, bbox, (255, 0, 255), 2)

        cv2.line(frame, (x, y), (x + l, y), colour, t)
        cv2.line(frame, (x, y), (x, y + l), colour, t)

        cv2.line(frame, (x1, y1), (x1 - l, y1), colour, t)
        cv2.line(frame, (x1, y1), (x1, y1 - l), colour, t)

        cv2.line(frame, (x1, y), (x1 - l, y), colour, t)
        cv2.line(frame, (x1, y), (x1, y + l), colour, t)

        cv2.line(frame, (x, y1), (x + l, y1), colour, t)
        cv2.line(frame, (x, y1), (x, y1 - l), colour, t)

        return frame


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    ptime = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame, bboxs = detector.findFaces(frame)

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (36, 178, 255), 3)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()