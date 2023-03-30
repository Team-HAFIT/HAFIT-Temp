# 파이썬 패키지 설치
# pip install mediapipe opencv-python

## 패키지 불러오기
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

## 각도 지정
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

## 비디오 출력
cap = cv2.VideoCapture(0)


# Curl counter variables [카운트 변수 지정]
counter = 0 
stage = None

## Setup mediapipe instance [mediapipe 인스턴스 설정(감지 신뢰도, 추적 신뢰도)]
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True) as pose:
    while cap.isOpened():
        ## 캡처를 읽어 드림
        ret, frame = cap.read()
        
         ## Recolor image to RGB [이미지를 RGB를 지정(초기값)]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR [RGB를 지정]
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks  [랜드마크 추출]
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates  [좌표 가져오기(이미지 참조)]
            lt_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lt_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            lt_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            rt_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            rt_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            rt_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angle [3값의 각도 계산]
            angle1 = calculate_angle(lt_shoulder, lt_elbow, lt_wrist)
            angle2 = calculate_angle(rt_shoulder, rt_elbow, rt_wrist)
            
              ## Visualize angle [각도 시각화]
            cv2.putText(image, str(int(angle1)),        # 문자로 각도를 표시
                           tuple(np.multiply(lt_elbow, [640, 480]).astype(int)),      # 각도를 표시할 위치
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA       # 각도를 표시할 텍스트 설정
                                 )
            cv2.putText(image, str(int(angle2)), 
                           tuple(np.multiply(rt_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                                 )
            ## Curl counter logic [카운트 논리]
            if angle1 > 160 and angle2 > 160:
                stage = "down"
            if angle1 < 30 and angle1 < 30 and stage =='down':
                stage="up"
                counter +=1
                print(counter)

            
        except:
            pass
        
        # Render curl counter [카운터 창 표시]
        # Setup status box  [나타낼 창 지정(파란색 창)]
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data [개수 카운트]
		# PutText (이미지, 입력할 글, 좌표, font, 글자크기, 글자색, 글자굵기,  
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data [상태 확인(UP, DOWN)]
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections => 랜더링 탐지 설정
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                #(BGR(색상)=>파랑,두께,원반경)=> 점
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                #(BGR(색상)=>파랑,두께,원반경)=> 선
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)             # 캡처 나타내기, 화면 이름 설정
        # 화면 종료 [cv2.waitKey(10) & 0xFF는 특정 키보드를 눌렀을때]
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()