import cv2
import mediapipe as mp
import os
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)   
cap.set(4, 480)  
cap.set(10, 150)  

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

pasttime = 0

folder = 'colors'
mylist = os.listdir(folder) if os.path.exists(folder) else []
overlist = []
col = [0, 0, 255]  

color_list = [
    [0, 0, 255],    
    [0, 255, 0],    
    [255, 0, 0],    
    [0, 255, 255],  
    [255, 255, 0],  
    [255, 0, 255]   
]
color_index = 0  
col = color_list[color_index]

for i in mylist:
    image = cv2.imread(os.path.join(folder, i))
    if image is not None:
        overlist.append(image)

if overlist:
    header = overlist[0]
else:
    header = np.zeros((100, 640, 3), np.uint8)
    print(" No valid images found in 'colors' folder. Using blank header.")

xp, yp = 0, 0
canvas = np.zeros((480, 640, 3), np.uint8)

def fingers_up(landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    if landmarks[tip_ids[0]][1] < landmarks[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if landmarks[tip_ids[id]][2] < landmarks[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    lanmark = []

    if results.multi_hand_landmarks:
        for hn in results.multi_hand_landmarks:
            for id, lm in enumerate(hn.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lanmark.append([id, cx, cy])
            mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)

    if len(lanmark) != 0:
        x1, y1 = lanmark[8][1], lanmark[8][2]
        x2, y2 = lanmark[12][1], lanmark[12][2]

        fingers = fingers_up(lanmark)

        if fingers == 0:
            canvas = np.zeros((480, 640, 3), np.uint8)
            print(" Canvas cleared with gesture!")

        elif fingers == 5:
            filename = f"canvas_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f" Canvas saved as {filename}. Exiting...")

            cv2.putText(frame, "Saved! Exiting...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('cam', frame)
            cv2.waitKey(1500)  
            
            break

        elif lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
            xp, yp = 0, 0
            print('Selection mode')

            if 1 <= fingers <= len(color_list):
                color_index = fingers - 1
                col = color_list[color_index]
                print(f" Color changed to: {col}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)

        elif lanmark[8][2] < lanmark[6][2]:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if col == [0, 0, 0]:
                cv2.line(frame, (xp, yp), (x1, y1), col, 100, cv2.FILLED)
                cv2.line(canvas, (xp, yp), (x1, y1), col, 100, cv2.FILLED)

            cv2.line(frame, (xp, yp), (x1, y1), col, 25, cv2.FILLED)
            cv2.line(canvas, (xp, yp), (x1, y1), col, 25, cv2.FILLED)
            print('Drawing mode')
            xp, yp = x1, y1

    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.zeros((480, 640, 3), np.uint8)
        print(" Canvas cleared by key!")

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    frame[0:100, 0:640] = header

    cv2.circle(frame, (30, 70), 20, col, cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - pasttime) if pasttime else 0
    pasttime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (490, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow('cam', frame)
    cv2.imshow('canvas', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
