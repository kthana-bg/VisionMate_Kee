# VisionMate: AI Eye Strain Monitor and Ergonomic Coach
# Using WebRTC for robust real-time video streaming

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import pandas as pd
from collections import deque

# MediaPipe tools for face and body tracking
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Dashboard configuration
st.set_page_config(page_title="VisionMate", layout="wide")

# Custom CSS styling for glassmorphism interface
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 100%); color: white; }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin-bottom: 20px;
    }
    .metric-value { 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #00f2fe; 
    }
    .metric-label {
        font-size: 0.9rem;
        color: #cccccc;
        margin-bottom: 0px;
    }
    .alert-warning {
        background: rgba(255, 100, 100, 0.2);
        border-left: 4px solid #ff6666;
        padding: 10px;
        border-radius: 8px;
    }
    .alert-success {
        background: rgba(100, 255, 100, 0.2);
        border-left: 4px solid #66ff66;
        padding: 10px;
        border-radius: 8px;
    }
    .alert-info {
        background: rgba(100, 100, 255, 0.2);
        border-left: 4px solid #6666ff;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'ear_history' not in st.session_state:
    st.session_state.ear_history = deque(maxlen=150)
    st.session_state.posture_history = deque(maxlen=150)
    st.session_state.blink_counter = 0
    st.session_state.last_ear = 0.35
    st.session_state.start_time = time.time()
    st.session_state.current_ear = 0.35
    st.session_state.current_posture = "Good Posture"
    st.session_state.fatigue_status = "Healthy"
    st.session_state.current_advice = {}
    st.session_state.blink_rate = 0

# Load the trained deep learning model
@st.cache_resource
def load_visionmate_model():
    try:
        model = tf.keras.models.load_model('visionmate_eye_MobileNetV2.keras')
        return model
    except Exception:
        st.info("Running in standard mode (deep learning model not found)")
        return None

# Eye Aspect Ratio calculation
def calculate_ear(landmarks, eye_indices):
    """Measures eye openness using Euclidean distance between landmarks"""
    # Vertical landmarks (top and bottom of eye)
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    
    # Horizontal landmarks (corners of eye)
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    
    vertical_dist = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal_dist = 2.0 * np.linalg.norm(p1 - p4)
    
    if horizontal_dist == 0:
        return 0.35
    
    return vertical_dist / horizontal_dist

# Posture analysis using shoulder alignment
def analyze_posture(pose_landmarks, frame_shape):
    """Detects slouching based on shoulder angle and head position"""
    if not pose_landmarks:
        return "Unknown", 1.0
    
    try:
        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]
        
        # Calculate shoulder angle
        shoulder_vec = np.array([right_shoulder.x - left_shoulder.x, 
                                 right_shoulder.y - left_shoulder.y])
        shoulder_angle = np.arctan2(shoulder_vec[1], shoulder_vec[0]) * 180 / np.pi
        
        # Calculate head offset from shoulder center
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        head_offset = abs(nose.x - shoulder_mid_x)
        
        if abs(shoulder_angle) > 15 or head_offset > 0.08:
            return "Slouching Detected", shoulder_angle
        else:
            return "Good Posture", shoulder_angle
    except:
        return "Unknown", 1.0

# Generate personalized ergonomic advice
def generate_advice(ear_value, fatigue_flag, posture_status, blink_rate):
    """Creates adaptive coaching messages"""
    if fatigue_flag:
        return {
            "title": "Eye Strain Alert",
            "message": "Blink rate decreasing. Follow the 20-20-20 rule: look 20 feet away for 20 seconds.",
            "type": "warning"
        }
    elif posture_status == "Slouching Detected":
        return {
            "title": "Posture Correction Needed",
            "message": "Straighten the back. Keep shoulders relaxed and aligned with ears.",
            "type": "warning"
        }
    elif blink_rate < 8 and blink_rate > 0:
        return {
            "title": "Take a Break",
            "message": "Eyes feel dry. Close them for a few seconds and look at something distant.",
            "type": "info"
        }
    else:
        return {
            "title": "Good Form",
            "message": "Ergonomic habits are within healthy range. Keep maintaining good posture.",
            "type": "success"
        }

# Video processor class for WebRTC streaming
class VisionMateProcessor(VideoProcessorBase):
    """Handles real-time video frame processing for eye strain and posture detection"""
    
    def __init__(self):
        # Initialize MediaPipe models
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Load AI model
        self.eye_model = load_visionmate_model()
        
        # Processing state
        self.last_inference_time = 0
        self.inference_interval = 0.15  # Process every 150ms for smooth performance
        
        # Eye indices for MediaPipe
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]
        
        # Metrics storage
        self.current_ear = 0.35
        self.current_posture = "Good Posture"
        self.fatigue_detected = False
        self.blink_counter = 0
        self.last_ear_value = 0.35
        
    def recv(self, frame):
        """Process each incoming video frame"""
        current_time = time.time()
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Throttle processing for performance
        if current_time - self.last_inference_time >= self.inference_interval:
            self.last_inference_time = current_time
            
            # Run face mesh and pose detection
            face_results = self.face_mesh.process(rgb_img)
            pose_results = self.pose_tracker.process(rgb_img)
            
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                
                # Calculate Eye Aspect Ratio
                left_ear = calculate_ear(landmarks, self.left_eye_idx)
                right_ear = calculate_ear(landmarks, self.right_eye_idx)
                avg_ear = (left_ear + right_ear) / 2.0
                self.current_ear = avg_ear
                
                # Blink detection
                if self.last_ear_value > 0.25 and avg_ear < 0.2:
                    self.blink_counter += 1
                self.last_ear_value = avg_ear
                
                # Calculate blink rate
                elapsed_time = current_time - st.session_state.start_time
                if elapsed_time > 0:
                    blink_rate = (self.blink_counter / elapsed_time) * 60
                    st.session_state.blink_rate = blink_rate
                
                # Deep learning inference for fatigue detection
                if self.eye_model is not None:
                    eye_roi = cv2.resize(rgb_img, (96, 96))
                    eye_roi = np.expand_dims(eye_roi, axis=0) / 255.0
                    prediction = self.eye_model.predict(eye_roi, verbose=0)
                    self.fatigue_detected = prediction[0][0] > 0.5
                else:
                    self.fatigue_detected = avg_ear < 0.22
                
                # Update session state for UI display
                st.session_state.current_ear = avg_ear
                st.session_state.fatigue_status = "Strain Detected" if self.fatigue_detected else "Healthy"
                st.session_state.ear_history.append(avg_ear)
                
                # Store blink counter in session state
                st.session_state.blink_counter = self.blink_counter
                
            else:
                # No face detected
                st.session_state.current_ear = 0
                st.session_state.fatigue_status = "No Face Detected"
            
            # Posture analysis
            if pose_results.pose_landmarks:
                posture_status, angle = analyze_posture(pose_results.pose_landmarks, img.shape)
                self.current_posture = posture_status
                st.session_state.current_posture = posture_status
                st.session_state.posture_history.append(0 if posture_status == "Good Posture" else 1)
            else:
                st.session_state.current_posture = "Detecting..."
            
            # Generate advice
            blink_rate = getattr(st.session_state, 'blink_rate', 0)
            advice = generate_advice(self.current_ear, self.fatigue_detected, 
                                     self.current_posture, blink_rate)
            st.session_state.current_advice = advice
        
        # Draw visual overlays on the frame
        h, w = img.shape[:2]
        
        # Status indicators
        cv2.rectangle(img, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (200, 80), (255, 255, 255), 1)
        cv2.putText(img, "VisionMate", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 255), 2)
        cv2.putText(img, f"EAR: {self.current_ear:.2f}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Color-coded feedback on frame
        if self.fatigue_detected:
            cv2.putText(img, "EYE STRAIN", (w - 120, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.current_ear > 0:
            cv2.putText(img, "EYES HEALTHY", (w - 130, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Posture feedback
        if self.current_posture == "Slouching Detected":
            cv2.putText(img, "SLUMPING", (w - 100, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif self.current_posture == "Good Posture":
            cv2.putText(img, "GOOD POSTURE", (w - 110, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main application
def main():
    st.title("VisionMate")
    st.caption("AI-Powered Eye Strain Monitor and Ergonomic Coach")
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("Controls")
        enable_analytics = st.checkbox("Enable Analytics Recording", value=True)
        st.markdown("---")
        st.header("Session Info")
        session_duration = time.time() - st.session_state.start_time
        st.metric("Session Duration", f"{int(session_duration // 60)}m {int(session_duration % 60)}s")
        st.metric("Total Blinks", st.session_state.blink_counter)
        
        # Display average ear for session
        if len(st.session_state.ear_history) > 10:
            avg_ear_session = np.mean(list(st.session_state.ear_history))
            st.metric("Average EAR", f"{avg_ear_session:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area - two columns
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="glass-card">Live Monitoring</div>', unsafe_allow_html=True)
        
        # WebRTC streamer
        ctx = webrtc_streamer(
            key="visionmate",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VisionMateProcessor,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]}
                ]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        
        if not ctx.state.playing:
            st.info("Click Start to begin real-time eye strain and posture monitoring")
    
    with col_right:
        # Real-time metrics display
        st.markdown('<div class="glass-card">Real-Time Metrics</div>', unsafe_allow_html=True)
        
        # Eye Aspect Ratio
        ear_value = st.session_state.current_ear
        if ear_value > 0.25:
            ear_color = "#00ff00"
        elif ear_value > 0:
            ear_color = "#ff6666"
        else:
            ear_color = "#cccccc"
        
        st.markdown('<p class="metric-label">Eye Aspect Ratio (EAR)</p>', unsafe_allow_html=True)
        
        if ear_value > 0:
            ear_display = f"{ear_value:.3f}"
        else:
            ear_display = "---"
        
        st.markdown(f'<p class="metric-value" style="color:{ear_color}">{ear_display}</p>', unsafe_allow_html=True)
        
        # Eye status
        fatigue_status = st.session_state.fatigue_status
        if fatigue_status == "Strain Detected":
            st.error("STRAIN DETECTED")
        elif fatigue_status == "Healthy":
            st.success("Eyes Healthy")
        elif fatigue_status == "No Face Detected":
            st.warning("Position face in frame")
        else:
            st.info("Monitoring")
        
        st.markdown("---")
        
        # Posture status
        posture = st.session_state.current_posture
        if posture == "Slouching Detected":
            st.error("Posture: Poor - Sit up straight")
        elif posture == "Good Posture":
            st.success("Posture: Good")
        else:
            st.info("Detecting posture")
        
        st.markdown("---")
        
        # ERGONOMIC COACH advice
        st.markdown('<div class="glass-card">Ergonomic Coach</div>', unsafe_allow_html=True)
        advice = st.session_state.current_advice
        if advice and advice.get('title'):
            if advice['type'] == 'warning':
                st.markdown(f'<div class="alert-warning"><strong>{advice["title"]}</strong><br>{advice["message"]}</div>', unsafe_allow_html=True)
            elif advice['type'] == 'success':
                st.markdown(f'<div class="alert-success"><strong>{advice["title"]}</strong><br>{advice["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-info"><strong>{advice["title"]}</strong><br>{advice["message"]}</div>', unsafe_allow_html=True)
    
    # Analytics section at bottom
    if enable_analytics:
        st.markdown('<div class="glass-card">Session Analytics</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Eye Aspect Ratio Trend")
            if len(st.session_state.ear_history) > 5:
                ear_df = pd.DataFrame(list(st.session_state.ear_history), columns=["EAR Value"])
                st.line_chart(ear_df, height=250)
            else:
                st.caption("Collecting data")
        
        with col_b:
            st.subheader("Posture Quality")
            if len(st.session_state.posture_history) > 5:
                posture_df = pd.DataFrame(list(st.session_state.posture_history), columns=["Poor Posture Events"])
                st.line_chart(posture_df, height=250)
            else:
                st.caption("Collecting data")
        
        # Summary statistics
        if len(st.session_state.ear_history) > 10:
            avg_ear = np.mean(list(st.session_state.ear_history))
            poor_posture = sum(st.session_state.posture_history)
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Average EAR", f"{avg_ear:.3f}")
            with col_s2:
                st.metric("Poor Posture Events", poor_posture)
            with col_s3:
                if st.session_state.posture_history:
                    compliance = 100 - (poor_posture / len(st.session_state.posture_history) * 100)
                else:
                    compliance = 100
                st.metric("Ergonomics Score", f"{compliance:.0f} percent")

if __name__ == "__main__":
    main()
