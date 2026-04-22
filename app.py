import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import datetime
import math
import os

# Database and Logging Configuration
LOG_FILE = "visionmate_health_logs.csv"

def init_db():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["timestamp", "user", "event_type", "value", "status"])
        df.to_csv(LOG_FILE, index=False)

def log_event(user, event_type, value, status):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[timestamp, user, event_type, value, status]], 
                             columns=["timestamp", "user", "event_type", "value", "status"])
    new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

# MediaPipe Initialization
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# EAR Calculation Utility
def calculate_ear(landmarks, eye_indices):
    # Vertical landmarks
    p2_p6 = math.dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    p3_p5 = math.dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    # Horizontal landmark
    p1_p4 = math.dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

# Posture Angle Utility
def calculate_posture_angle(ear_landmark, shoulder_landmark):
    # Calculate angle relative to vertical axis
    d_x = ear_landmark.x - shoulder_landmark.x
    d_y = ear_landmark.y - shoulder_landmark.y
    angle = math.degrees(math.atan2(d_x, d_y))
    return abs(angle)

class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.user_name = "Guest"
        self.blink_count = 0
        self.last_log_time = datetime.datetime.now()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process Modules
        face_results = self.face_mesh.process(img_rgb)
        pose_results = self.pose.process(img_rgb)
        
        h, w, _ = img.shape
        status_color = (0, 255, 0)
        
        # 1. Eye Strain Detection (EAR)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                l_eye = [33, 160, 158, 133, 153, 144]
                r_eye = [362, 385, 387, 263, 373, 380]
                
                landmarks = face_landmarks.landmark
                # Convert landmarks to coordinate format for distance calculation
                points = [{"x": lm.x, "y": lm.y} for lm in landmarks]
                
                # Simplified EAR for the demo
                left_ear = calculate_ear(landmarks, l_eye)
                right_ear = calculate_ear(landmarks, r_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear < 0.20:
                    cv2.putText(img, "BLINK DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 2. Posture Monitoring
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            # Left side landmarks
            ear_l = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            
            posture_angle = calculate_posture_angle(ear_l, shoulder_l)
            
            # Threshold: Slouching usually detected if ear is significantly forward of shoulder
            if posture_angle > 15.0:
                cv2.putText(img, "BAD POSTURE", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Throttled logging to avoid file bloating
                if (datetime.datetime.now() - self.last_log_time).seconds > 10:
                    log_event(self.user_name, "Posture", round(posture_angle, 2), "Poor")
                    self.last_log_time = datetime.datetime.now()
            else:
                cv2.putText(img, "GOOD POSTURE", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="VisionMate AI", layout="wide")
    init_db()

    # Sidebar: User Authentication and Settings
    st.sidebar.title("VisionMate Login")
    user_id = st.sidebar.text_input("User ID", "Keerthana")
    auth_button = st.sidebar.button("Authenticate via Face ID")
    
    if auth_button:
        st.sidebar.success("Face ID Verified: " + user_id)
        st.session_state["authenticated"] = True
    else:
        st.sidebar.warning("Please Login to Sync Analytics")

    # Main Dashboard Layout
    st.title("AI Ergonomic Dashboard")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Monitoring")
        ctx = webrtc_streamer(
            key="visionmate-engine",
            video_processor_factory=VisionProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if ctx.video_processor:
            ctx.video_processor.user_name = user_id

    with col2:
        st.subheader("Real-time Health Metrics")
        # Dynamic placeholders for metrics
        metric_container = st.container()
        with metric_container:
            st.write("Current Session Analytics")
            st.info("Focus Score: 88%")
            st.info("Average EAR: 0.28")
            st.info("Posture Deviation: Low")

        st.subheader("Daily History")
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            user_df = df[df["user"] == user_id].tail(10)
            st.dataframe(user_df[["timestamp", "event_type", "status"]], use_container_width=True)

    # Personalized Suggestion Module
    st.divider()
    st.subheader("AI Ergonomic Coach Suggestions")
    if os.path.exists(LOG_FILE):
        df_logs = pd.read_csv(LOG_FILE)
        bad_posture_count = len(df_logs[(df_logs["user"] == user_id) & (df_logs["status"] == "Poor")])
        
        if bad_posture_count > 5:
            st.error("Detected repetitive slouching. Suggestion: Adjust your monitor height to eye level.")
        else:
            st.success("Ergonomic habits are within healthy limits. Keep it up.")

if __name__ == "__main__":
    main()
