import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time
import yaml
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading
from datetime import datetime

from database_manager import DatabaseManager
from model_comparator import ModelComparator
from utils.face_auth import FaceAuthenticator

# Configuration
try:
    st.set_page_config(page_title="VisionMate", layout="wide", initial_sidebar_state="collapsed")
except:
    pass

@st.cache_resource
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}

CONFIG = load_config()

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
.main-header { font-size: 2.2rem; font-weight: 700; text-align: center; 
    background: linear-gradient(135deg, #fff, #a0a0ff); -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
.live-badge { background: #f44336; color: white; padding: 4px 12px; border-radius: 20px; 
    font-size: 0.8rem; animation: pulse 1s infinite; display: inline-block; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
.status-live { background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); 
    border-radius: 20px; padding: 2rem; border: 1px solid rgba(255,255,255,0.2); 
    text-align: center; height: 200px; }
.status-good { color: #4caf50; font-size: 3rem; font-weight: bold; }
.status-warning { color: #ff9800; font-size: 3rem; font-weight: bold; animation: pulse 1s infinite; }
.status-danger { color: #f44336; font-size: 3rem; font-weight: bold; animation: pulse 0.8s infinite; }
.video-container { position: relative; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_state():
    defaults = {
        'initialized': True,
        'logged_in': False,
        'user_id': None,
        'user_name': None,
        'session_id': None,
        'db': DatabaseManager(),
        'auth': FaceAuthenticator(),
        'comparator': None,
        'live_eye_status': 'NORMAL',
        'live_posture_status': 'GOOD',
        'live_health': 50,
        'live_running': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# REAL-TIME LIVE VIDEO PROCESSOR
class LiveVideoProcessor(VideoProcessorBase):
    type_ = "code"
    
    def __init__(self):
        self.frame_count = 0
        self.last_process_time = 0
        self.process_interval = 0.5
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()
            
            # Real-time processing every 0.5 seconds
            if (current_time - self.last_process_time > self.process_interval and 
                st.session_state.get('comparator')):
                self.last_process_time = current_time
                try:
                    results = st.session_state.comparator.process_frame(img.copy())
                    
                    # Update LIVE status immediately
                    eye_result = results.get('eye', {}).get('C1', {})
                    posture_result = results.get('posture', {}).get('C2', {})
                    
                    st.session_state.live_eye_status = eye_result.get('classification', 'NORMAL')
                    st.session_state.live_posture_status = posture_result.get('status', 'GOOD')
                    st.session_state.live_health = results.get('health_score', 50)
                    
                    # Database logging
                    if st.session_state.get('session_id') and st.session_state.get('db'):
                        st.session_state.db.log_model_comparison(
                            st.session_state.session_id, 
                            st.session_state.user_id, 
                            results
                        )
                        
                except Exception as e:
                    pass
            
            # Draw real-time overlay on video
            self._draw_live_overlay(img)
            
        except Exception:
            pass
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_live_overlay(self, img):
        try:
            # Get current live status
            health = st.session_state.get('live_health', 50)
            eye_status = st.session_state.get('live_eye_status', 'NORMAL')
            posture_status = st.session_state.get('live_posture_status', 'GOOD')
            
            h, w = img.shape[:2]
            
            # Top header
            cv2.rectangle(img, (10, 10), (w-10, 70), (20,20,20), -1)
            cv2.putText(img, "REAL-TIME ANALYSIS", (25, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            # Health progress bar
            cv2.rectangle(img, (20, 85), (w-20, 115), (60,60,60), -1)
            health_width = int((health / 100.0) * (w - 40))
            health_color = (0,255,0) if health > 70 else (0,165,255) if health > 40 else (0,0,255)
            cv2.rectangle(img, (20, 85), (20 + health_width, 115), health_color, -1)
            cv2.putText(img, f"Health Score: {int(health)}%", (30, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # Eye status
            y_pos = 130
            eye_color = (0,255,0) if eye_status == "NORMAL" else (0,0,255)
            cv2.rectangle(img, (20, y_pos), (400, y_pos + 50), (20,20,20), -1)
            cv2.putText(img, f"EYE STRAIN: {eye_status}", (30, y_pos + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, eye_color, 2)
            
            # Posture status
            y_pos2 = y_pos + 60
            posture_color = (0,255,0) if posture_status == "GOOD" else (0,0,255)
            cv2.rectangle(img, (20, y_pos2), (450, y_pos2 + 50), (20,20,20), -1)
            cv2.putText(img, f"POSTURE: {posture_status}", (30, y_pos2 + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, posture_color, 2)
            
            # Live indicator
            cv2.circle(img, (w-50, 50), 15, (0,255,0), -1)
            cv2.putText(img, "LIVE", (w-75, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
        except Exception:
            pass

# Login/Register Page
def show_login_register_page():
    st.markdown('<div class="main-header">VisionMate</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:rgba(255,255,255,0.7);">Real-time Eye Strain and Posture Coach</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown("#### Login")
        login_frame = st.camera_input("Look at camera for login", key="login_camera")
        
        if st.button("Login with Face", type="primary", use_container_width=True):
            if login_frame is not None:
                try:
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
                            st.rerun()
                        else:
                            st.error("Face not recognized. Please register first.")
                    else:
                        st.error("No face detected")
                except Exception:
                    st.error("Login processing error")
            else:
                st.warning("Please capture a face image")
    
    with col_right:
        st.markdown("#### Create Account")
        new_user_name = st.text_input("Enter your full name", placeholder="Full Name")
        reg_frame = st.camera_input("Capture face image", key="reg_camera")
        
        if st.button("Complete Registration", type="primary", use_container_width=True):
            if new_user_name and reg_frame is not None:
                try:
                    bytes_data = reg_frame.getvalue()
                    nparr = np.frombuffer(bytes_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    embedding = st.session_state.auth.extract_embedding(frame)
                    if embedding is not None:
                        reduced = st.session_state.auth.reduce_dimension(embedding)
                        user_id = st.session_state.db.create_user(new_user_name, reduced.tolist())
                        st.success("Registration successful! You can now login.")
                    else:
                        st.error("Failed to extract face features from image")
                except Exception:
                    st.error("Registration processing error")
            else:
                st.warning("Please enter name and capture face image")

# Live Monitor Page - TRUE REAL-TIME
def live_monitor_page():
    st.markdown("### Real-Time Live Monitoring")
    st.markdown('<span class="live-badge">LIVE</span> <span style="color:rgba(255,255,255,0.8);">Continuous tracking - updates every 0.5 seconds</span>', unsafe_allow_html=True)
    
    # 3-column layout: Live Video + Eye Status + Posture Status
    col_video, col_eye, col_posture = st.columns([2, 1, 1])
    
    with col_video:
        st.markdown("**Live Video Feed**")
        
        # RTC configuration for reliable WebRTC
        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        })
        
        # Start REAL-TIME WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="live_video_processor",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=LiveVideoProcessor,
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
            st.success("Live tracking active - Real-time analysis running")
            st.caption("Video shows live overlay with health score and status")
        else:
            st.info("Click START button above to begin real-time tracking")
    
    # LIVE EYE STRAIN STATUS CARD
    with col_eye:
        st.markdown("**Live Eye Strain Status**")
        eye_status = st.session_state.get('live_eye_status', 'NORMAL')
        status_class = "status-good" if eye_status == "NORMAL" else "status-danger"
        
        st.markdown(f'''
        <div class="status-live">
            <div class="{status_class}">EYE</div>
            <div style="font-size:1.3rem; margin-top:15px; color:rgba(255,255,255,0.95); font-weight:500;">
                {eye_status}
            </div>
            <div style="font-size:0.85rem; color:rgba(255,255,255,0.6); margin-top:15px;">
                Updates every 0.5s
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # LIVE POSTURE STATUS CARD
    with col_posture:
        st.markdown("**Live Posture Status**")
        posture_status = st.session_state.get('live_posture_status', 'GOOD')
        status_class = "status-good" if posture_status == "GOOD" else "status-danger"
        
        st.markdown(f'''
        <div class="status-live">
            <div class="{status_class}">POSTURE</div>
            <div style="font-size:1.3rem; margin-top:15px; color:rgba(255,255,255,0.95); font-weight:500;">
                {posture_status}
            </div>
            <div style="font-size:0.85rem; color:rgba(255,255,255,0.6); margin-top:15px;">
                Updates every 0.5s
            </div>
        </div>
        ''', unsafe_allow_html=True)

# Analytics Page
def analytics_page():
    st.markdown("### Analytics Dashboard")
    
    hours = st.selectbox("Select time range", [1, 6, 12, 24, 48, 72], 
                        index=3, format_func=lambda x: f"Last {x} hours")
    
    try:
        stats = st.session_state.db.get_strain_statistics(st.session_state.user_id, hours=hours)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Eye Strain Events", stats['eye_strain_count'])
        col2.metric("Poor Posture Events", stats['posture_poor_count'])
        col3.metric("Eye Strain Rate", f"{stats['eye_strain_percentage']:.1f}%")
        col4.metric("Poor Posture Rate", f"{stats['posture_poor_percentage']:.1f}%")
        
        st.divider()
        
        df = st.session_state.db.get_user_analytics(st.session_state.user_id, hours=hours)
        if not df.empty:
            fig_eye = px.line(df, x='timestamp', y='eye_score', 
                            title='Eye Strain Trend Over Time',
                            labels={'eye_score': 'Strain Score', 'timestamp': 'Time'})
            fig_eye.add_hline(y=0.5, line_dash="dash", line_color="red")
            fig_eye.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_eye, use_container_width=True)
            
            fig_posture = px.line(df, x='timestamp', y='posture_score', 
                                title='Posture Score Over Time',
                                labels={'posture_score': 'Slouch Score', 'timestamp': 'Time'})
            fig_posture.add_hline(y=0.6, line_dash="dash", line_color="red")
            fig_posture.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_posture, use_container_width=True)
            
            if st.button("Export Data (CSV)", type="primary"):
                csv_data = df.to_csv(index=False)
                st.download_button("Download CSV", csv_data, 
                                 f"visionmate_{st.session_state.user_name}_{hours}h.csv", "text/csv")
        else:
            st.info("No analytics data yet. Use live monitoring to generate data.")
            
    except Exception:
        st.error("Analytics temporarily unavailable")

# Main Dashboard
def show_main_dashboard():
    # Header
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown('<div class="main-header">VisionMate Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:rgba(255,255,255,0.7);">Real-time Health Monitoring</div>', unsafe_allow_html=True)
    
    with col_header2:
        if st.button("Logout", use_container_width=True):
            if st.session_state.session_id:
                st.session_state.db.end_session(st.session_state.session_id)
            # Reset session
            st.session_state.update({
                'logged_in': False,
                'user_id': None,
                'user_name': None,
                'session_id': None,
                'comparator': None,
                'live_eye_status': 'NORMAL',
                'live_posture_status': 'GOOD',
                'live_health': 50
            })
            st.rerun()
    
    st.divider()
    
    # Session info
    if st.session_state.session_id:
        try:
            session_info = st.session_state.db.get_current_session(st.session_state.session_id)
            if session_info:
                duration = session_info.get('duration_minutes', 0)
                st.caption(f"Active session: {duration} minutes | User: {st.session_state.user_name}")
        except:
            pass
    
    # Tabs
    tab_live, tab_analytics = st.tabs(["Live Monitor", "Analytics"])
    
    with tab_live:
        live_monitor_page()
    
    with tab_analytics:
        analytics_page()

# MAIN APPLICATION
def main():
    if st.session_state.get('logged_in') and st.session_state.get('comparator'):
        show_main_dashboard()
    else:
        show_login_register_page()

if __name__ == "__main__":
    main()
