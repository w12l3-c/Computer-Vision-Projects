import cv2
import mediapipe as mp
import time

# Face Detection
mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection(0.8)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ptime = 0

while cap.isOpened():
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face.process(frameRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(frame, detection)
            # print(f"{id},{detection}: {detection.score}")
            # print(detection.location_data.relative_bounding_box)
            bound = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bound.xmin * iw), int(bound.ymin * ih), int(bound.width * iw), int(bound.height * ih)
            cv2.rectangle(frame, bbox, (255, 0, 255), 2)
            cv2.putText(frame, f'Human {id+1}: {int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (97, 56, 232), 2)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (36, 178, 255), 3)
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()