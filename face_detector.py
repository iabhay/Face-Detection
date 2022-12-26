import cv2
from random import randrange


tracked_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
webcam =  cv2.VideoCapture(0)
while True:
    successful_frame_read, frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cordinates = tracked_face.detectMultiScale(grayscaled_img)
    for (x,y,w,h) in face_cordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (randrange(256),randrange(256),randrange(256)), 10)

    cv2.imshow('RaDiumZ Face Detector', frame)
    key = cv2.waitKey(1)
    if key==81 or key == 113:
        break
    
    
webcam.release()
print("Code Completed")