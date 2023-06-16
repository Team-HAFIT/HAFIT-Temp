import cv2
import mediapipe as mp
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLBoxItem
import pyqtgraph as pg
import threading
import queue

def rotate_x(angle, landmarks):                 # 랜드마크의 각도 조절
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle), -np.sin(angle)],
                                [0, np.sin(angle), np.cos(angle)]])
    return np.dot(landmarks, rotation_matrix)

def check_knee_forward(world_landmarks):        # 무릎이 앞으로 갔는지 확인
    left_hip = world_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    left_knee = world_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = world_landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]

    right_hip = world_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = world_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = world_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

    left_knee_forward = left_knee[0] > left_ankle[0]  
    right_knee_forward = right_knee[0] > right_ankle[0]  

    return left_knee_forward, right_knee_forward


def calculate_knee_angle(world_landmarks, frame):    #무릎의 각도를 확인
    left_hip = world_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    left_knee = world_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = world_landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
    left_thigh = left_knee - left_hip
    left_calf = left_ankle - left_knee
    left_knee_angle = np.degrees(np.arccos(np.dot(left_thigh, left_calf) / (np.linalg.norm(left_thigh) * np.linalg.norm(left_calf))))

    right_hip = world_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = world_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = world_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
    right_thigh = right_knee - right_hip
    right_calf = right_ankle - right_knee
    right_knee_angle = np.degrees(np.arccos(np.dot(right_thigh, right_calf) / (np.linalg.norm(right_thigh) * np.linalg.norm(right_calf))))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    thickness = 2

    left_knee_angle_text = "Left Knee Angle: {:.2f}".format(left_knee_angle)
    right_knee_angle_text = "Right Knee Angle: {:.2f}".format(right_knee_angle)
    cv2.putText(frame, left_knee_angle_text, (20, 50), font, fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, right_knee_angle_text, (20, 80), font, fontScale, (0, 0, 255), thickness, cv2.LINE_AA)

    return left_knee_angle, right_knee_angle

def draw_landmarks_custom(frame, landmarks, connections, left_knee_forward, right_knee_forward,
                          left_knee_size, right_knee_size,
                          landmark_drawing_spec=mp.solutions.drawing_styles.DrawingSpec(),
                          connection_drawing_spec=mp.solutions.drawing_styles.DrawingSpec()):
    for idx, landmark in enumerate(landmarks):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

        if idx == mp.solutions.pose.PoseLandmark.LEFT_KNEE.value:
            size = left_knee_size
            if left_knee_forward:
                color = (0, 0, 255)  # green
            else:
                color = (255, 255, 255) # red
        elif idx == mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value:
            size = right_knee_size
            if right_knee_forward:
                color = (255, 255, 255)  # green
            else:
                color = (0, 0, 255)  # red
        else:
            size = landmark_drawing_spec.circle_radius
            color = landmark_drawing_spec.color

        # Increase the size of the landmarks
        frame = cv2.circle(frame, (x, y), int(size), color, landmark_drawing_spec.thickness)

    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if landmarks[start_idx].visibility < 0 or landmarks[end_idx].visibility < 0 or landmarks[start_idx].presence < 0 or landmarks[end_idx].presence < 0:
                continue

            x_start, y_start = int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0])
            x_end, y_end = int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0])

            frame = cv2.line(frame, (x_start, y_start), (x_end, y_end), connection_drawing_spec.color, connection_drawing_spec.thickness)

    return frame



def graph_update_thread(landmark_queue):
    app = QtWidgets.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('3D Pose Visualization')
    w.setCameraPosition(distance=3, elevation=20, azimuth=-90)  # Update camera position and angles

    scatter = gl.GLScatterPlotItem(pos=np.zeros((33, 3), dtype=np.float32), color=(0, 0, 0, 255), size=12)
    w.addItem(scatter)

    # Add lines for pose connections
    lines = []
    for connection in mp.solutions.pose.POSE_CONNECTIONS:
        line = gl.GLLinePlotItem(width=3, color=(0, 30, 0, 255))
        w.addItem(line)
        lines.append(line)

    rotation_angle = np.radians(90)  # 90 degrees in radians
    camera_azimuth = -90  # initial camera azimuth

    while True:
        try:
            world_landmarks = landmark_queue.get(block=True, timeout=1)
            rotated_landmarks = rotate_x(rotation_angle, world_landmarks)  # Rotate landmarks
            scatter.setData(pos=rotated_landmarks)  # Set rotated landmarks

            # Update pose connection lines
            for i, connection in enumerate(mp.solutions.pose.POSE_CONNECTIONS):
                start, end = connection
                lines[i].setData(pos=np.vstack([rotated_landmarks[start], rotated_landmarks[end]]))

            QtWidgets.QApplication.processEvents()

        except queue.Empty:
            pass



def video_processing_thread(video_path, landmark_queue):
    mp_pose = mp.solutions.pose.Pose()
    cap = cv2.VideoCapture(0)

    left_knee_size = 6
    right_knee_size = 6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_pose.process(image)

        if result.pose_landmarks:
            # Check if knees are forward
            world_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result.pose_world_landmarks.landmark])
            left_knee_forward, right_knee_forward = check_knee_forward(world_landmarks)

            # Adjust landmark size based on knee position
            if left_knee_forward:
                left_knee_size = 12
            else:
                left_knee_size = 6

            if right_knee_forward:
                right_knee_size = 6
            else:
                right_knee_size = 12

            knee_colors = {
                mp.solutions.pose.PoseLandmark.LEFT_KNEE.value: (0, 0, 255) if left_knee_forward else (255, 255, 255),
                mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value: (255, 255, 255) if right_knee_forward else (0, 0, 255)
            }

            # Draw pose landmarks on the frame
            frame = draw_landmarks_custom(
                frame, result.pose_landmarks.landmark, mp.solutions.pose.POSE_CONNECTIONS, left_knee_forward, right_knee_forward,
                left_knee_size, right_knee_size,
                landmark_drawing_spec=mp.solutions.drawing_styles.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                connection_drawing_spec=mp.solutions.drawing_styles.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Put world landmarks in the queue for visualization
            landmark_queue.put(world_landmarks)

            # Calculate knee angles and display on frame
            calculate_knee_angle(world_landmarks, frame)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = "video.mp4"
    landmark_queue = queue.Queue()

    video_thread = threading.Thread(target=video_processing_thread, args=(video_path, landmark_queue))
    graph_thread = threading.Thread(target=graph_update_thread, args=(landmark_queue,))

    video_thread.start()
    graph_thread.start()

    video_thread.join()
    graph_thread.join()

if __name__ == "__main__":
    main()
