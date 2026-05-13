# VisionMate - AI Eye Strain Monitor and Ergonomic Coach
# Faculty of Artificial Intelligence and Cyber Security, UTeM

import os
# Force CPU mode for MediaPipe to avoid GPU errors in headless environment
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import yaml
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

from database_manager import DatabaseManager
from model_comparator import ModelComparator
from utils.face_auth import FaceAuthenticator

# Load configuration
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}

CONFIG = load_config()

# Page configuration
st.set_page_config(
    page_title="VisionMate - AI Eye Strain and Ergonomic Coach",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# INITIALIZE ALL SESSION STATE VARIABLES
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.session_id = None
    st.session_state.db = DatabaseManager()
    st.session_state.auth = FaceAuthenticator()
    st.session_state.comparator = None
    st.session_state.eye_strain_count = 0
    st.session_state.posture_poor_count = 0
    st.session_state.last_log_time = 0
    st.session_state.last_alert_time = 0

# Glassmorphism CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #fff, #a0a0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 2rem;
        font-size: 0.9rem;
    }
    
    .status-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .status-normal {
        color: #4caf50;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .status-danger {
        color: #f44336;
        font-size: 2rem;
        font-weight: bold;
        animation: pulse 0.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .stAlert {
        background: rgba(0,0,0,0.6);
        backdrop-filter: blur(8px);
    }
    
    .live-badge {
        background-color: #f44336;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        display: inline-block;
        animation: pulse 1s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# THREAD-SAFE VIDEO PROCESSOR FOR WEBRTC
# NO st.session_state ACCESS INSIDE THIS CLASS
# ============================================================================

class VisionMateVideoProcessor(VideoProcessorBase):
    """
    Processes video frames in real-time using WebRTC.
    IMPORTANT: This runs in a separate thread.
    DO NOT access st.session_state inside this class.
    """
    
    def __init__(self, comparator, user_id, session_id, db):
        self.comparator = comparator
        self.user_id = user_id
        self.session_id = session_id
        self.db = db
        
        # Thread-safe internal storage (not shared with Streamlit)
        self.frame_count = 0
        self.last_process_time = 0
        self.last_log_time = 0
        self.process_interval = 0.5
        self.log_interval = 5.0
        
        # Store latest results internally (thread-safe)
        self.latest_results = None
        self.alert_eye = False
        self.alert_posture = False
        self.last_alert_time = 0
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Called for each video frame from the camera"""
        self.frame_count += 1
        current_time = time.time()
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame at reduced frequency
        if current_time - self.last_process_time >= self.process_interval:
            self.last_process_time = current_time
            
            if self.comparator is not None:
                try:
                    # Run detection
                    results = self.comparator.process_frame(img)
                    
                    # Store results internally (not in session_state)
                    self.latest_results = results
                    
                    # Get custom model results
                    eye_result = results['eye'].get('C1', {})
                    posture_result = results['posture'].get('C2', {})
                    
                    eye_strained = eye_result.get('classification') == 'STRAINED'
                    posture_poor = posture_result.get('status') == 'SLOUCHING'
                    
                    # Update alerts with cooldown
                    if current_time - self.last_alert_time > 10:
                        self.alert_eye = eye_strained
                        self.alert_posture = posture_poor
                        if eye_strained or posture_poor:
                            self.last_alert_time = current_time
                    
                    # Log to database periodically
                    if current_time - self.last_log_time >= self.log_interval:
                        self.db.log_model_comparison(self.session_id, self.user_id, results)
                        self.last_log_time = current_time
                        
                except Exception as e:
                    print(f"Processing error: {e}")
        
        # Draw overlay using internal variables (NOT session_state)
        img = self._draw_overlay(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_overlay(self, img):
        """Draw status overlay on video frame using internal state only"""
        try:
            if self.latest_results:
                health = self.latest_results.get('health_score', 50)
                
                if health >= 70:
                    color = (0, 255, 0)
                elif health >= 40:
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)
                
                # Header
                cv2.rectangle(img, (0, 0), (200, 35), (0, 0, 0), -1)
                cv2.putText(img, "VisionMate", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Health score
                cv2.rectangle(img, (0, 40), (180, 75), (0, 0, 0), -1)
                cv2.putText(img, f"Health: {health}", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Eye strain alert
                if self.alert_eye:
                    cv2.rectangle(img, (0, 80), (250, 115), (0, 0, 0), -1)
                    cv2.putText(img, "ALERT: Eye Strain!", (10, 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Posture alert
                if self.alert_posture:
                    cv2.rectangle(img, (0, 120), (260, 155), (0, 0, 0), -1)
                    cv2.putText(img, "ALERT: Poor Posture!", (10, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        except Exception as e:
            print(f"Overlay error: {e}")
        
        return img
    
    def get_latest_results(self):
        """Thread-safe method to get latest results (called from UI thread)"""
        return self.latest_results
    
    def get_alerts(self):
        """Thread-safe method to get alert status"""
        return self.alert_eye, self.alert_posture


# ============================================================================
# LOGIN AND REGISTER PAGE
# ============================================================================

def show_login_register_page():
    st.markdown('<div class="main-header">VisionMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI Eye Strain Monitor and Ergonomic Coach</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2, gap="large")
    
    # LOGIN SECTION
    with col_left:
        with st.container():
            st.markdown("#### Login")
            login_frame = st.camera_input("Look at the camera", key="login_camera", label_visibility="collapsed")
            
            if st.button("Login with Face", type="primary", width='stretch'):
                if login_frame is not None:
                    bytes_data = login_frame.getvalue()
                    nparr = np.frombuffer(bytes_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    embedding = st.session_state.auth.extract_embedding(frame)
                    
                    if embedding is not None:
                        reduced = st.session_state.auth.reduce_dimension(embedding)
                        users = st.session_state.db.get_all_users()
                        
                        best_match = None
                        best_score = 0
                        
                        for user in users:
                            stored = np.array(user['face_embedding'])
                            similarity = st.session_state.auth.compare_embeddings(reduced, stored)
                            
                            if similarity > best_score and similarity > 0.75:
                                best_score = similarity
                                best_match = user
                        
                        if best_match:
                            st.session_state.logged_in = True
                            st.session_state.user_id = best_match['user_id']
                            st.session_state.user_name = best_match['user_name']
                            st.session_state.session_id = st.session_state.db.start_session(best_match['user_id'])
                            st.session_state.comparator = ModelComparator(CONFIG)
                            st.session_state.last_alert_time = time.time()
                            st.rerun()
                        else:
                            st.error("Face not recognized. Please register first.")
                    else:
                        st.error("No face detected")
                else:
                    st.warning("Please capture a face image")
    
    # REGISTER SECTION
    with col_right:
        with st.container():
            st.markdown("#### Create Account")
            new_user_name = st.text_input("Enter your full name", placeholder="e.g., John Doe")
            
            st.markdown("Capture 5 face images from different angles")
            
            reg_frames = []
            reg_cols = st.columns(5)
            
            for i in range(5):
                with reg_cols[i]:
                    cap = st.camera_input(f"{i+1}", key=f"reg_cam_{i}", label_visibility="collapsed")
                    if cap is not None:
                        bytes_data = cap.getvalue()
                        nparr = np.frombuffer(bytes_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        reg_frames.append(frame)
                        st.success("OK")
            
            if st.button("Complete Registration", type="primary", width='stretch'):
                if new_user_name and len(reg_frames) >= 3:
                    embeddings = []
                    for frame in reg_frames:
                        emb = st.session_state.auth.extract_embedding(frame)
                        if emb is not None:
                            embeddings.append(st.session_state.auth.reduce_dimension(emb))
                    
                    if len(embeddings) >= 3:
                        avg_embedding = np.mean(embeddings, axis=0).tolist()
                        user_id = st.session_state.db.create_user(new_user_name, avg_embedding)
                        st.success("Registration successful! You can now login.")
                        st.balloons()
                    else:
                        st.error("Failed to extract face features")
                else:
                    st.warning("Please enter name and capture at least 3 face images")


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def show_dashboard():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">VisionMate</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI Eye Strain and Posture Monitor</div>', unsafe_allow_html=True)
    
    # Logout button
    col_right, col_left = st.columns([5, 1])
    with col_left:
        if st.button("Logout", width='stretch'):
            st.session_state.db.end_session(st.session_state.session_id)
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.session_state.session_id = None
            st.session_state.comparator = None
            st.rerun()
    
    st.divider()
    
    # Session info
    session_info = st.session_state.db.get_current_session(st.session_state.session_id)
    if session_info:
        duration = session_info.get('duration_minutes', 0)
        st.caption(f"Session active for {duration} minutes | User: {st.session_state.user_name}")
    
    # Live status badge
    st.markdown('<span class="live-badge">LIVE</span> Real-time monitoring active', unsafe_allow_html=True)
    st.caption("Camera feed with low-latency WebRTC. Results update continuously.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Live Monitor", "Analytics"])
    
    # ========================================================================
    # TAB 1: LIVE MONITOR
    # ========================================================================
    
    with tab1:
        st.markdown("### Real-Time Ergonomic Monitoring")
        st.caption("Your camera is being analyzed in real-time for eye strain and posture")
        
        # ================================================================
        # VIDEO PROCESSOR CONTAINER
        # We'll use a placeholder to display results from the processor
        # ================================================================
        
        # Cache values locally BEFORE lambda to avoid session_state access in thread
        comparator = st.session_state.comparator
        user_id = st.session_state.user_id
        session_id = st.session_state.session_id
        db = st.session_state.db
        
        # Placeholder for processor results (will be updated via a different mechanism)
        results_placeholder = st.empty()
        status_col1, status_col2, status_col3 = st.columns(3)
        
        if comparator is not None:
            # Start WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="visionmate-webrtc",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda c=comparator, uid=user_id, sid=session_id, d=db: 
                    VisionMateVideoProcessor(c, uid, sid, d),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 30}
                    },
                    "audio": False
                },
                async_processing=True,
            )
            
            if webrtc_ctx.state.playing:
                st.success("Camera is active - Real-time monitoring in progress")
                
                # Note: Since we cannot access processor's internal state directly,
                # we need a different approach. For WebRTC, the overlay is drawn
                # directly on the video feed, so we don't need separate status cards.
                # The video feed itself shows the health score and alerts.
                
                st.info("Health score and alerts are displayed directly on the video feed above.")
                
            else:
                st.info("Click 'Start' to begin monitoring")
        else:
            st.info("Please wait... System initializing")
        
        st.divider()
        
        # Quick tips
        st.markdown("#### Quick Tips for Healthy Computing")
        tips_col1, tips_col2, tips_col3 = st.columns(3)
        
        with tips_col1:
            st.markdown("**20-20-20 Rule**")
            st.caption("Every 20 minutes, look at something 20 feet away for 20 seconds")
        
        with tips_col2:
            st.markdown("**Ergonomic Setup**")
            st.caption("Top of screen at eye level, back supported, feet flat on floor")
        
        with tips_col3:
            st.markdown("**Take Breaks**")
            st.caption("Stand up and stretch every 30-60 minutes")
    
    # ========================================================================
    # TAB 2: ANALYTICS
    # ========================================================================
    
    with tab2:
        st.markdown("### My Ergonomic Analytics")
        st.caption("Your personal behavior patterns over time")
        
        hours = st.selectbox("Time Range", [1, 6, 12, 24, 48, 72], index=3, format_func=lambda x: f"Last {x} hours")
        
        stats = st.session_state.db.get_strain_statistics(st.session_state.user_id, hours=hours)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Eye Strain Events", stats['eye_strain_count'])
        with col2:
            st.metric("Poor Posture Events", stats['posture_poor_count'])
        with col3:
            st.metric("Eye Strain Rate", f"{stats['eye_strain_percentage']:.1f}%")
        with col4:
            st.metric("Poor Posture Rate", f"{stats['posture_poor_percentage']:.1f}%")
        
        st.divider()
        
        df = st.session_state.db.get_user_analytics(st.session_state.user_id, hours=hours)
        
        if not df.empty:
            st.markdown("#### Eye Strain Trend (Model C1 - My Custom CNN)")
            
            fig_eye = px.line(df, x='timestamp', y='eye_score',
                              title='Eye Fatigue Score Over Time',
                              labels={'eye_score': 'Fatigue Score (0=Alert, 1=Strained)', 'timestamp': 'Time'})
            fig_eye.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Strain Threshold")
            fig_eye.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=400)
            st.plotly_chart(fig_eye, use_container_width=True)
            
            st.markdown("#### Posture Trend (Model C2 - My Custom LSTM)")
            
            fig_posture = px.line(df, x='timestamp', y='posture_score',
                                   title='Slouching Probability Over Time',
                                   labels={'posture_score': 'Slouching Probability (0=Good, 1=Poor)', 'timestamp': 'Time'})
            fig_posture.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="Slouching Threshold")
            fig_posture.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=400)
            st.plotly_chart(fig_posture, use_container_width=True)
            
            st.divider()
            st.markdown("#### AI-Powered Insights")
            
            avg_eye = df['eye_score'].mean()
            avg_posture = df['posture_score'].mean()
            
            if avg_eye > 0.6:
                st.warning("Your average eye strain level is high. Consider more frequent breaks using the 20-20-20 rule.")
            elif avg_eye > 0.4:
                st.info("Your average eye strain level is moderate. Monitor your screen time and blink regularly.")
            else:
                st.success("Your eye health is good. Keep maintaining your current habits!")
            
            if avg_posture > 0.6:
                st.warning("You slouch frequently. Try adjusting your chair height and monitor position.")
            elif avg_posture > 0.4:
                st.info("Your posture shows occasional slouching. Set hourly reminders to check your posture.")
            else:
                st.success("Your posture is excellent! Keep up the good work.")
            
            st.divider()
            if st.button("Export My Data (CSV)", type="primary"):
                export_df = st.session_state.db.export_all_logs(st.session_state.user_id)
                if not export_df.empty:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"visionmate_data_{st.session_state.user_name}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Not enough data yet. Continue using VisionMate to see your analytics.")
    
    st.divider()
    st.caption("VisionMate - AI Eye Strain Monitor and Ergonomic Coach")
    st.caption("Faculty of Artificial Intelligence and Cyber Security, Universiti Teknikal Malaysia Melaka")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    if st.session_state.logged_in and st.session_state.comparator is not None:
        show_dashboard()
    else:
        show_login_register_page()


if __name__ == "__main__":
    main()
    