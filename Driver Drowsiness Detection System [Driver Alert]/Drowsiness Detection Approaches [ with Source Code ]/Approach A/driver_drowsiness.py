#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
#Pygame for playing the Alarming Sound
from pygame import mixer
#os for dealing with the Directory
import os

#Initilizing the Sound
mixer.init()
sound = mixer.Sound('alarm.wav')
path = os.getcwd()


#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)
ratio = 0
score = 0
thicc = 0

#Function for Computing Distance between two Points
def compute(ptA,ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

#Blinked Function to calculate the Ratio ofthe Individual Eyes
def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    #Checking if it is blinked
    if(ratio>0.25):
        return 2
    elif(ratio>0.18 and ratio<=0.25):
        return 1
    else:
        return 0


#While Loop Running for each Frame
while True:
    face_frame, frame = cap.read()
    height,width = frame.shape[:2]
    
    #Converting the Image to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
            landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
            landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        #Now judge what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy=0
            active=0
            score += 1
            if(sleep>6):
                status="SLEEPING !!!"
                color = (0,0,255)
                cv2.putText(frame, "Take a Break...", (10,height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

        elif(left_blink==1 or right_blink==1):
            sleep=0
            active=0
            drowsy+=1
            score += 1
            if(drowsy>4):
                status="DROWSY !"
                color = (255,0,0)

        else:
            drowsy=0
            sleep=0
            active+=1
            score -= 3
            if(active>9):
                status="ACTIVE"
                color = (0,255,0)
        
        #To put the Text on the Top Right of the Frame
        cv2.putText(frame, status, (width-200,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)


        #For creating circles on the Face Landmarks
        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
            
    
    #For Dealing with the Scores
    if(score<0):
        score = 0   
    cv2.putText(frame,'Score:'+str(score),(10,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>12):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        
        #For Creating a Red Border during the detection of Sleepy State
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 


    #For Creating the Windows
    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
          break