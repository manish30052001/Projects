import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade - Copy.xml')

lbl=['Close','Open','Yawn','No Yawn']
framespersecond = 0
model = load_model('Kaggle Saved Models/cnnCat2_new_100_epochs_SF-1_044_grayscale.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
print("The total number of frames in this video is ", framespersecond)
font = cv2.FONT_HERSHEY_DUPLEX 
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
fpred=[99]

while(True):
    ret, frame= cap.read()
    frame1 = frame
    height,width = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    faces = face.detectMultiScale(frame,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(frame)
    right_eye =  reye.detectMultiScale(frame)

    cv2.rectangle(frame1, (0,height-100) , (155,height) , (0,0,0) , thickness=cv2.FILLED )
    cv2.rectangle(frame1, (width-150,0) , (width,50) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame1, (x,y) , (x+w,y+h) , (0,255,0) , 2 )
        face_new = frame[y:y+h,x:x+w]
        count = count+1
        #r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        face_new = cv2.resize(face_new,(145,145))
        face_new = face_new/255
        #r_eye=  r_eye.reshape(145,145,-1)
        face_new =  face_new.reshape(145,145,-1)
        face_new = np.expand_dims(face_new,axis=0)
        #rpred = model.predict_classes(r_eye)
        fpred = (model.predict(face_new) > 0.5).astype("int32")
        #print(rpred[0][0])
        #if(rpred[0]==1):
        if((fpred[0][0]==1).any()):
            lbl='Yawn' 
        #f(rpred[0]==0):
        if((fpred[0][1]==1).any()):
            lbl='No Yawn'
        break

    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame1, (x,y) , (x+w,y+h) , (0,255,0) , 1 )
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        #r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(145,145))
        r_eye= r_eye/255
        #r_eye=  r_eye.reshape(145,145,-1)
        r_eye=  r_eye.reshape(145,145,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        #rpred = model.predict_classes(r_eye)
        rpred = (model.predict(r_eye) > 0.5).astype("int32")
        #print(rpred[0][0])
        #if(rpred[0]==1):
        if((rpred[0][3]==1).any()):
            lbl='Open' 
        #f(rpred[0]==0):
        if((rpred[0][2]==1).any()):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame1, (x,y) , (x+w,y+h) , (0,255,0) , 1 )
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        #l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(145,145))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(145,145,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        #lpred = model.predict_classes(l_eye)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        lpred = (model.predict(l_eye) > 0.5).astype("int32")
        #print(lpred[0][0])
        #if((lpred[0]==1):
        if((lpred[0][3]==1).any()):
            lbl='Open'   
        if((lpred[0][2]==1).any()):
        #if(lpred[0]==0):
            lbl='Closed'
        break

    if(((rpred[0][2]==1).any()) and ((lpred[0][2]==1).any())):
    #if(rpred[0]==0 and lpred[0]==0):
        score = score + 1
        cv2.putText(frame1,"Closed",(10,height-60), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    elif(((rpred[0][3]==1).any()) and ((lpred[0][3]==1).any())):
        score = score - 1
        cv2.putText(frame1,"Open",(10,height-60), font, 1,(255,255,255),1,cv2.LINE_AA) 
    if((fpred[0][0]==1).any()):
        score = score + 1
        cv2.putText(frame1,"Yawn",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    #if(rpred[0]==0):
    elif((fpred[0][1]==1).any()):
        score = score - 1
        cv2.putText(frame1,"No Yawn",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame1,'Score:'+str(score),(width-145,30), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>2):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame1)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<8):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame1,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()