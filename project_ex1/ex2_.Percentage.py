import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture(0) 

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img)                    ## true일때 전체, false일때 그부분만 
    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        # Right Knee
        angle1 = detector.findAngle(img, 23, 25, 27)
        # Left Knee
        angle2 = detector.findAngle(img, 24, 26, 28)

        per = np.interp(angle1+angle2, (540, 360), (100, 0)) ## 값, 100일때. 80
        bar = np.interp(angle1+angle2, (550, 360), (100, 650))

        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:                          # 퍼센트(%)
            color = (0, 255, 0)
            if dir == 0:                        # 위치값
                count += 0.5                    # 개수
                dir = 1                         
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)

        # Draw Curl Count
        cv2.rectangle(img,(0,600),(120,720), (0,255,0),cv2.FILLED)
        cv2.putText(img, str(int(count)),(30,670),cv2.FONT_HERSHEY_PLAIN,5,
                    (255,0,0),5)

    

    cv2.imshow('Raw Webcam Feed', img)   # 프레임을 보여줌, 프레임에 제목을 설정

    if cv2.waitKey(10) & 0xFF == ord('q'):           # 캠 끄기 설정
        break

cap.release()
cv2.destroyAllWindows()