import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define the scaling factor for the landmarks
LANDMARK_SCALE_FACTOR = 1

# Open the exercise video file
cap = cv2.VideoCapture('path/to/exercise/video.mp4')

# Create windows for the video and landmarks
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.namedWindow('Landmarks', cv2.WINDOW_NORMAL)

# Set the positions of the windows
cv2.moveWindow('Video', 0, 0)
cv2.moveWindow('Landmarks', 1600, 0)

# Define the sizes of the windows
VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720
LANDMARK_WIDTH, LANDMARK_HEIGHT = 400, 400

# Resize the landmark window
landmark_frame = np.zeros((LANDMARK_HEIGHT, LANDMARK_WIDTH, 3), np.uint8)

# Create a Pose object for pose estimation
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the pose estimation model
        results = pose.process(frame_rgb)

        # Draw the landmarks on the landmark frame
        landmark_frame.fill(0)
        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                landmark_frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS, 
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=4, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2))

        # Resize the video frame to the desired size
        video_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        # Show the video and landmark frames
        cv2.imshow('Video', video_frame)
        cv2.imshow('Landmarks', landmark_frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
