import mediapipe as mp
import cv2

print("MediaPipe version:", mp.__version__)

# Test face detection
mp_face_detection = mp.solutions.face_detection
print("Face detection available:", mp_face_detection is not None)

# Test face mesh
mp_face_mesh = mp.solutions.face_mesh
print("Face mesh available:", mp_face_mesh is not None)

print("MediaPipe is working correctly!")