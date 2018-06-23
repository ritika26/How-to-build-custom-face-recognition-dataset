import cv2
import numpy as np

# Loading the cascades

cascade_face= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eye= cv2.CascadeClassifier('haarcascade_eye.xml')
cascade_smile =cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray,originalimg):
    faces= cascade_face.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(originalimg,(x,y),(x+w,y+h),(0,0,255),3)
        roi_gray= gray[y:y+h,x:x+w]
        roi_color= originalimg[y:y+h,x:x+w]
        eyes = cascade_eye.detectMultiScale(roi_gray,1.1,3)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles= cascade_smile.detectMultiScale(roi_gray,1.1,2)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)
    return originalimg # We return the image with the detector rectangles.
# Doing some face recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _,originalimg= video_capture.read()
    gray= cv2.cvtColor(originalimg,cv2.COLOR_BGR2GRAY)
    canvas= detect(gray,originalimg)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() #Turning the  the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.



