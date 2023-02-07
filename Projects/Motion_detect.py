import cv2
import time
import numpy
import pandas as pd
import datetime

first_frame = None
status_lst = [None, None]
times_lst = []
df = pd.DataFrame(columns = ["start", "end"])       #Dataframe to store time values when detect object or see movement

video = cv2.VideoCapture(0)

run = True
counter = 1
while run:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)      #convert gray scale to Gaussian Blur

    if counter == 1:
        if first_frame is None:  # storing the first frame
            first_frame = gray
    elif counter == 2:
        if first_frame is None:  # storing the first frame
            first_frame = gray


    delta_frame = cv2.absdiff(first_frame, gray)        #difference between first and other frames
    thres = cv2.threshold(delta_frame, 160, 255, cv2.THRESH_BINARY)[1]      #the threshold value, difference less than 30 convert to black, higher convert to white
    thres = cv2.dilate(thres, None, iterations = 2)
    (cnts,_) = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     #define contour areas(border)

    for contours in cnts:
        if cv2.contourArea(contours) < 1000:        #remove noise and shadows, only keep white for parts greater than 1000 pixels
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contours)       #adding rectangle around the object
        frame = cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 3)
    status_lst.append(status)       #list of status every frame
    status_lst = status_lst[-2:]

    #when their is a change of status bewteen last two status, means the object move, record in the time list
    if status_lst[-1] == 1 and status_lst[-2] == 0:
        times_lst.append(datetime.datetime.now())
    if status_lst[-1] == 0 and status_lst[-2] == 1:
        times_lst.append(datetime.datetime.now())

    cv2.imshow('frame', frame)
    cv2.imshow('Capturing', gray)
    cv2.imshow('Delta Frame', delta_frame)
    cv2.imshow('Threshold', thres)

    key = cv2.waitKey(1)  # generate a new frame after one millisecond
    counter += 1
    if key == ord('q'):  # when q button is pressed the loop stops
        run = False

print(status_lst)
print(times_lst)
for i in range(0, len(times_lst)):
    df = df.append({"start":times_lst[i], "end":times_lst[i]}, ignore_index = True) #Store time value into dataframe

print(df)
video.release()
cv2.destroyAllWindows()