import cv2 as cv
import random
import numpy as np
import pandas as pd
import os
import uuid
import re
from PIL import Image
import urllib
import urllib.request

# ------------------ Common cv2 Function ------------------
# read image and store image
img = cv.imread("Projects/images/FlW8a5zXwAA2GOM.jpeg")

# convert image to grayscale
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# change image size
resized_img = cv.resize(img, dsize=(500, 500), interpolation=cv.INTER_CUBIC)

# show image
cv.imshow("read_img", img)
cv.imshow("grey_img", grey_img)
cv.imshow("resized_img", resized_img)

print(f"image: {img.shape}")
print(f"grey image: {grey_img.shape}")
print(f"resized image: {resized_img.shape}")
# ------------------ Common cv2 Function ------------------


# ------------------ Drawing things on image ------------------
# coordinates of the image
x, y, w, h = img.shape[1]//2-100, img.shape[0]//2-100, 200, 200

# draw rectangle in img
cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2, lineType=cv.LINE_8)

# draw circle in img
cv.circle(img, center=(x+w//2, y+h//2), radius=100, color=(0, 0, 255), thickness=2, lineType=cv.LINE_8)
cv.imshow("bounded_image", img)

# cv.putText write text
# ------------------ Drawing things on image ------------------


# ------------------ Face Recognition with Webcam ------------------
# bounding face function
def face_detect_demo(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("/Users/wallacelee/Downloads/haar_cascade/haarcascade_frontalface_default.xml")
    # scaleFactor everytime reduce the size of the image
    # minNeighbours is the number of times it needs to detect the face everytime
    # minSize and maxSize is the range of the face size
    # scaleFactor=1.05, minNeighbors=3, flags=0, minSize=(25, 25), maxSize=(200, 200)
    faces = face_detector.detectMultiScale(grey)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        # putting text on the Rectangle
        cv.putText(img, 'Human', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv.imshow("face_detect", img)
    cv.waitKey(1)


# read human image
img2 = cv.imread("Projects/images/stringquartet_1583901_1000.jpeg")
img3 = cv.imread("Projects/images/solvay.jpeg")
face_detect_demo(img2)
face_detect_demo(img3)

# read video
cap = cv.VideoCapture(0)    # 0 is the camera number

while True:
    ret, frame = cap.read()
    if not ret:
        break
    face_detect_demo(frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()   # release the camera
# ------------------ Face recognition with webcam ------------------


# ------------------ Screen Capturing ------------------
cap2 = cv.VideoCapture(0)
count = 1

while cap2.isOpened():
    ret_flag, frame = cap2.read()
    face_detect_demo(frame)

    if cv.waitKey(1) & 0xFF == ord("s"):
        path = os.path.join("Screencapture", f"{count}.screenshot.jpg")
        print(path)
        cv.imwrite(path, frame)
        print(f"save {count} image\n")
        count += 1

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap2.release()
# ------------------ Screen Capturing ------------------


# ------------------ Common functions ------------------
# wait for key press, both way works, prefer the first one
while True:
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    if ord("q") == cv.waitKey(0):
        break

# save image
cv.imwrite("Projects/images/grey_img.jpg", grey_img)

# 0 will wait forever
cv.waitKey(0)

# destroy all windows
cv.destroyAllWindows()
# ------------------ Common functions ------------------


# ------------------ Face model train ------------------
# Train image through database images
def getImageAndLabel(path):
    # save the face arrays
    faceSamples = []
    # save the names
    ids = []
    # save image information
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # detector
    face_detector = cv.CascadeClassifier("/Users/wallacelee/Downloads/haar_cascade/haarcascade_frontalface_alt2.xml")
    # loop through the image to store their information
    for imagePath in imagePaths:
        # convert to grayscale in PIL, with 1, L, P, RGB, RGBA, CMYK, YCbCr, I, F as the 9 modes
        PIL_img = Image.open(imagePath).convert("L")
        # convert picture to numpy array
        img_numpy = np.array(PIL_img, "uint8")
        # get the face information
        faces = face_detector.detectMultiScale(img_numpy)
        id = int(re.split('. ,!_', os.path.split(imagePath)[-1])[0])
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    print(f"ids: {ids}")
    print(f"faceSamples: {faceSamples}")
    return faceSamples, ids


if __name__ == "__main__":
    # picture path
    path = "Screencapture"
    # function to get the images and label data
    faces, ids = getImageAndLabel(path)
    # create recognizer
    recognizer = cv.face.LBPHFaceRecognizer_create()
    # train recognizer
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer.yml
    recognizer.write("trainer/trainer.yml")

    # Print the numer of faces trained and end program
    print(f"{len(np.unique(ids))} faces trained. Exiting Program")
# ------------------ Face model train ------------------


# ------------------ Face recognition ------------------
# create recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()
# load the trained data
recognizer.read("trainer/trainer.yml")

names = []

# ------------------ CyberSecurity ------------------
# They are using an API so I am not using this section
# The following code is safe precaution
warningtime = 0

# md5 hash function
def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf-8"))
    return m.hexdigest()

# text feedback base on the API
statusStr = {
    '0': 'Unlocked',
    '1': 'Locked',
    '2': 'Warning',
}

# warning function - use a sms or other types of api
def warning():
    smsapi = "https://smsapi.free-mobile.fr/sendmsg?user=USER_ID&pass=USER_KEY&msg=Warning!%20Someone%20is%20trying%20to%20break%20into%20your%20house!"
    user = 'USER_ID'
    password = md5('USER_KEY')
    content = 'Warning! Someone is trying to break into your house!'
    phone = 'PHONE_NUMBER'

    # send the warning message
    data = urllib.parse.urlencode({'user': user, 'pass': password, 'msg': content, 'phone': phone})
    send_url = smsapi + 'sms?' + data
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    print(statusStr[the_page])

# i will be just using text message in terminal
def warning2():
    print("Warning! An Unknown person is trying to enter!")
# ------------------ CyberSecurity ------------------

# prepare picture for recognition
def face_recognition(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("/Users/wallacelee/Downloads/haar_cascade/haarcascade_frontalface_alt2.xml")
    face = face_detector.detectMultiScale(gray)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
        cv.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)
        ids, conf = recognizer.predict(gray[y:y+h, x:x+w])
        print(f"ids: {ids}, conf: {conf}")
        if conf > 80:
            global warningtime
            warningtime += 1
            if warningtime > 100:
                warning2()
                warningtime = 0
            cv.putText(img, "Unknown", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
            cv.putText(img, str(names[ids-1]), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv.imshow("result", img)


def name(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        name = re.split('. ,!_', os.path.split(imagePath)[-1])[1]
        names.append(name)


cap = cv.VideoCapture(0)
name("Screencapture")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    face_recognition(frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

# ------------------ Face recognition with Video ------------------
class CaptureVideo(object):
    def net_video(self):
        # pick one video path from online
        cam = cv.VideoCapture('rtmp://ipaddress/livetv/cctv5')
        while cam.isOpened():
            ret, frame = cam.read()
            cv.imshow("Network", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    capture_video = CaptureVideo()
    capture_video.net_video()
# ------------------ Face recognition with Video ------------------

# ------------------ Face recognition ------------------

