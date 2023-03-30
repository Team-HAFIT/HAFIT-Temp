import cv2
import numpy as np
import time
import PoseModule as pm
import datetime

cap = cv2.VideoCapture(0)

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0

# Create a datetime object for the current time
now = datetime.datetime.now()

# Create a VideoWriter object with the current time in the filename
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video/output_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.mp4', fourcc, 30, (1280, 720))

# Create a boolean variable for recording
recording = False

try:
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            angle1 = detector.findAngle(img, 27, 25, 23)
            angle2 = detector.findAngle(img, 24, 26, 28)

            per = np.interp(angle1+angle2, (220, 340), (100, 0))
            bar = np.interp(angle1+angle2, (220, 340), (100, 650))

            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0

            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)

            cv2.rectangle(img,(0,600),(120,720), (0,255,0),cv2.FILLED)
            cv2.putText(img, str(int(count)),(30,670),cv2.FONT_HERSHEY_PLAIN,5,
                        (255,0,0),5)

        # Check if the 'r' key is pressed to start or stop recording
        if cv2.waitKey(1) & 0xFF == ord('r'):
            recording = not recording

        # Write each frame to the output video file if recording is True
        if recording:
            out.write(img)

        cv2.imshow('Raw Webcam Feed', img)

        # Check if the window is closed to stop the loop and release the VideoWriter object
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the VideoWriter object and destroy all windows
    out.release()
    cv2.destroyAllWindows()
