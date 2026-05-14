import os
# Force CPU mode for MediaPipe to avoid GPU errors in headless environment
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

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

# ============================================================================
# CONFIGURATION (Called only once)
# ============================================================================

# Page config - MUST be called only ONCE at the very top
try:
    st.set_page_config(
        page_title="VisionMate - AI Eye Strain and Ergonomic Coach",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except:
    pass  # Ignore if already called

# Load configuration
@st.cache_resource
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}

CONFIG = load_config()

# Glassmorphism CSS (Safe to call multiple times)
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
# INITIALIZE SESSION STATE (Safe initialization)
# ============================================================================

def init_session_state():
    defaults = {
        'initialized': True,
        'logged_in': False,
        'user_id': None,
        'user_name': None,
        'session_id': None,
        'db': None,
        'auth': None,
        'comparator': None,
        'last_analysis': None,
        'analysis_count': 0,
        'eye_strain_count': 0,
        'posture_poor_count': 0,
        'last_log_time': 0,
        'last_alert_time': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            if key == 'db':
                st.session_state[key] = DatabaseManager()
            elif key == 'auth':
                st.session_state[key] = FaceAuthenticator()
            else:
                st.session_state[key] = value

init_session_state()

class SafeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_process_time = 0
        self.process_interval = 1.0
        self.last_log_time = 0
        self.log_interval = 10.0
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()
            
            if current_time - self.last_process_time >= self.process_interval:
                self.last_process_time = current_time
                
                # Safe access to session state
                if st.session_state.get('comparator'):
                    try:
                        results = st.session_state.comparator.process_frame(img.copy())
                        st.session_state.last_analysis = {
                            'time': current_time,
                            'results': results,
                            'health_score': results.get('health_score', 50)
                        }
                    except:
                        pass
            
            img = self._draw_overlay(img)
            
        except:
            pass
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_overlay(self, img):
        try:
            if st.session_state.get('last_analysis'):
                health = st.session_state.last_analysis.get('health_score', 50)
                color = (0,255,0) if health >= 70 else (0,165,255) if health >= 40 else (0,0,255)
                
                cv2.rectangle(img, (0, 0), (200, 35), (0, 0, 0), -1)
                cv2.putText(img, "VisionMate", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.rectangle(img, (0, 40), (180, 75), (0, 0, 0), -1)
                cv2.putText(img, f"Health: {int(health)}%", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            pass
        return img

# ============================================================================
# LOGIN AND REGISTER PAGE
# ============================================================================

def show_login_register_page():
    st.markdown('<div class="main-header">VisionMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI Eye Strain Monitor and Ergonomic Coach</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown("#### Login")
        login_frame = st.camera_input("Look at the camera", key="login_camera")
        
        if st.button("Login with Face", type="primary", use_container_width=True):
            if login_frame:
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
                except Exception as e:
                    st.error("Login error")
            else:
                st.warning("Please capture image")
    
    with col_right:
        st.markdown("#### Create Account")
        new_user_name = st.text_input("Enter your full name")
        reg_frame = st.camera_input("Face image", key="reg_camera")
        
        if st.button("Complete Registration", type="primary", use_container_width=True):
            if new_user_name and reg_frame:
                try:
                    bytes_data = reg_frame.getvalue()
                    nparr = np.frombuffer(bytes_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    embedding = st.session_state.auth.extract_embedding(frame)
                    if embedding is not None:
                        reduced = st.session_state.auth.reduce_dimension(embedding)
                        user_id = st.session_state.db.create_user(new_user_name, reduced.tolist())
                        st.success("Registration successful!")
                    else:
                        st.error("Failed to extract face features")
                except Exception:
                    st.error("Registration error")
            else:
                st.warning("Enter name and capture image")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def show_dashboard():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">VisionMate</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI Eye Strain and Posture Monitor</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("Logout", use_container_width=True):
            if st.session_state.session_id:
                st.session_state.db.end_session(st.session_state.session_id)
            for key in ['logged_in', 'user_id', 'user_name', 'session_id', 'comparator', 'last_analysis']:
                st.session_state[key] = None if key != 'logged_in' else False
            st.rerun()
    
    st.divider()
    
    if st.session_state.session_id:
        try:
            session_info = st.session_state.db.get_current_session(st.session_state.session_id)
            if session_info:
                duration = session_info.get('duration_minutes', 0)
                st.caption(f"Session: {duration}min | User: {st.session_state.user_name}")
        except:
            pass
    
    st.markdown('<span class="live-badge">LIVE</span> Real-time monitoring', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Live Monitor", "Analytics"])
    
    with tab1:
        st.markdown("### Real-Time Monitoring")
        
        if not st.session_state.comparator:
            st.warning("Initializing...")
            return
        
        # WebRTC with full error protection
        try:
            webrtc_ctx = webrtc_streamer(
                key="webrtc_main",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=SafeVideoProcessor,
                media_stream_constraints={
                    "video": {"width": 640, "height": 480, "frameRate": 10},
                    "audio": False
                },
                async_processing=True,
            )
            
            if webrtc_ctx.state.playing:
                st.success("Live analysis active")
                
                if st.session_state.last_analysis:
                    health = st.session_state.last_analysis.get('health_score', 50)
                    results = st.session_state.last_analysis.get('results', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="status-card">', unsafe_allow_html=True)
                        status_class = "status-normal" if health >= 60 else "status-danger"
                        st.markdown(f'<div class="{status_class}"> {health:.0f}% </div>', unsafe_allow_html=True)
                        st.markdown('<div style="color:rgba(255,255,255,0.7);">Health</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        eye_status = results.get('eye', {}).get('C1', {}).get('classification', 'NORMAL')
                        st.markdown('<div class="status-card">', unsafe_allow_html=True)
                        status_class = "status-danger" if eye_status == 'STRAINED' else "status-normal"
                        st.markdown(f'<div class="{status_class}">Eye</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        posture_status = results.get('posture', {}).get('C2', {}).get('status', 'GOOD')
                        st.markdown('<div class="status-card">', unsafe_allow_html=True)
                        status_class = "status-danger" if posture_status == 'SLOUCHING' else "status-normal"
                        st.markdown(f'<div class="{status_class}">Posture</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Click START for live monitoring")
                
        except:
            st.warning("Using fallback analysis")
            fallback_img = st.camera_input("Analyze this photo")
            if fallback_img and st.session_state.comparator:
                with st.spinner("Analyzing..."):
                    try:
                        bytes_data = fallback_img.getvalue()
                        nparr = np.frombuffer(bytes_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        results = st.session_state.comparator.process_frame(frame)
                        st.session_state.last_analysis = {'results': results, 'health_score': results.get('health_score', 50)}
                        st.success("Analysis complete")
                        st.rerun()
                    except:
                        st.error("Analysis failed")
    
    with tab2:
        st.markdown("### Analytics")
        hours = st.selectbox("Time Range", [1, 6, 12, 24, 48, 72], index=3)
        
        try:
            stats = st.session_state.db.get_strain_statistics(st.session_state.user_id, hours)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Eye Strain Events", stats['eye_strain_count'])
            col2.metric("Poor Posture Events", stats['posture_poor_count'])
            col3.metric("Eye Strain Rate", f"{stats['eye_strain_percentage']:.1f}%")
            col4.metric("Poor Posture Rate", f"{stats['posture_poor_percentage']:.1f}%")
            
            df = st.session_state.db.get_user_analytics(st.session_state.user_id, hours)
            if not df.empty:
                fig_eye = px.line(df, x='timestamp', y='eye_score', title='Eye Strain Trend')
                fig_eye.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_eye, use_container_width=True)
                
                fig_posture = px.line(df, x='timestamp', y='posture_score', title='Posture Trend')
                fig_posture.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_posture, use_container_width=True)
                
                if st.button("Export Data"):
                    csv = df.to_csv()
                    st.download_button("Download", csv, "data.csv", "text/csv")
            else:
                st.info("No data yet")
        except:
            st.error("Analytics unavailable")

# ============================================================================
# MAIN (No page_config here)
# ============================================================================

def main():
    if st.session_state.get('logged_in') and st.session_state.get('comparator'):
        show_dashboard()
    else:
        show_login_register_page()

if __name__ == "__main__":
    main()
