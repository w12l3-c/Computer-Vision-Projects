import mediapipe as mp
import cv2
import time

# Face Mesh
mpMesh = mp.solutions.face_mesh
mesh = mpMesh.FaceMesh(max_num_faces=3, min_detection_confidence=0.75, min_tracking_confidence=0.75)   # max_num_faces, min_detection_confidence, min_tracking_confidence
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ptime = 0

while cap.isOpened():
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mesh.process(frameRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            ballSpec = mpDraw.DrawingSpec(color=(255, 186, 226), thickness=1, circle_radius=1)  # balls
            connectSpec = mpDraw.DrawingSpec(color=(157, 78, 13), thickness=1, circle_radius=1)  # connections
            mpDraw.draw_landmarks(frame, faceLms, mpMesh.FACEMESH_TESSELATION, landmark_drawing_spec=ballSpec, connection_drawing_spec=connectSpec)
            for id, lm in enumerate(faceLms.landmark):
                print(lm)
                ih, iw, ic = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                #print(id, x, y)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (36, 178, 255), 3)
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()