import cv2
import mediapipe as mp
import numpy as np, cv2 as cv;
import time
import tensorflow


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpFace = mp.solutions.face_mesh
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
face = mpFace.FaceMesh()


counter = 0
pTime = 0
cTime = 0


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    if results is True:
        counter = counter+1

    print(results.multi_hand_landmarks)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    frame_rate = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(frame_rate)), (10,70), cv2.FONT_HERSHEY_PLAIN,2,
                        (000,000,000), 2)
    cv2.putText(img, 'Counter: ' + str(counter), (10,170), cv2.FONT_HERSHEY_PLAIN, 1, (000,000, 000), 1)
    if frame_rate <= 20:
        cv2.putText(img, 'LOW FPS', (10,130), cv2.FONT_HERSHEY_PLAIN,1,(000,000,000), 1)
    else:
        cv2.putText(img, 'GOOD FPS', (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (000, 000, 000), 1)
    cv2.imshow("PhoraTECH", img)
    cv2.waitKey(1)


