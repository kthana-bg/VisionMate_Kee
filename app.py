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

# Load configuration
@st.cache_resource
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
    st.session_state.last_analysis = None
    st.session_state.analysis_count = 0
    st.session_state.eye_strain_count = 0
    st.session_state.posture_poor_count = 0
    st.session_state.last_log_time = 0
    st.session_state.last_alert_time = 0

class SafeVideoProcessor(VideoProcessorBase):
    """
    Cloud-safe video processor with minimal session_state access
    """
    def __init__(self, comparator=None, user_id=None, session_id=None, db=None):
        self.comparator = comparator
        self.user_id = user_id
        self.session_id = session_id
        self.db = db
        self.last_process_time = 0
        self.process_interval = 1.0
        self.last_log_time = 0
        self.log_interval = 10.0
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()
            
            # Process at reduced frequency for stability
            if current_time - self.last_process_time >= self.process_interval:
                self.last_process_time = current_time
                
                if self.comparator:
                    try:
                        results = self.comparator.process_frame(img.copy())
                        
                        # Update session state safely
                        if 'last_analysis' not in st.session_state:
                            st.session_state.last_analysis = {}
                        st.session_state.last_analysis = {
                            'time': current_time,
                            'results': results,
                            'health_score': results.get('health_score', 50)
                        }
                        
                        # Log periodically
                        if current_time - self.last_log_time >= self.log_interval:
                            if self.db and self.session_id:
                                self.db.log_model_comparison(self.session_id, self.user_id, results)
                            self.last_log_time = current_time
                            
                    except Exception as e:
                        pass  # Silent fail for cloud stability
            
            # Draw overlay
            img = self._draw_overlay(img)
            
        except Exception:
            pass
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_overlay(self, img):
        try:
            # Safe overlay drawing
            if hasattr(st.session_state, 'last_analysis') and st.session_state.last_analysis:
                health = st.session_state.last_analysis.get('health_score', 50)
                
                if health >= 70:
                    color = (0, 255, 0)
                elif health >= 40:
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)
                
                cv2.rectangle(img, (0, 0), (200, 35), (0, 0, 0), -1)
                cv2.putText(img, "VisionMate", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.rectangle(img, (0, 40), (180, 75), (0, 0, 0), -1)
                cv2.putText(img, f"Health: {int(health)}%", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            pass
        
        return img

# ============================================================================
# LOGIN AND REGISTER PAGE (IMPROVED)
# ============================================================================

def show_login_register_page():
    st.markdown('<div class="main-header">VisionMate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI Eye Strain Monitor and Ergonomic Coach</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2, gap="large")
    
    # LOGIN SECTION
    with col_left:
        st.markdown("#### Login")
        login_frame = st.camera_input("Look at the camera", key="login_camera", label_visibility="collapsed")
        
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
                            st.session_state.last_alert_time = time.time()
                            st.rerun()
                        else:
                            st.error("Face not recognized. Please register first.")
                    else:
                        st.error("No face detected")
                except Exception as e:
                    st.error("Login processing error")
            else:
                st.warning("Please capture a face image")
    
    # REGISTER SECTION
    with col_right:
        st.markdown("#### Create Account")
        new_user_name = st.text_input("Enter your full name", placeholder="")
        
        st.markdown("Capture face image")
        
        reg_frame = st.camera_input("Face image", key="reg_camera", label_visibility="collapsed")
        
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
                        st.error("Failed to extract face features")
                except Exception:
                    st.error("Registration processing error")
            else:
                st.warning("Please enter name and capture face image")

# ============================================================================
# MAIN DASHBOARD (CLOUD-SAFE)
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
        if st.button("Logout", use_container_width=True):
            if st.session_state.session_id:
                st.session_state.db.end_session(st.session_state.session_id)
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.session_state.session_id = None
            st.session_state.comparator = None
            st.session_state.last_analysis = None
            st.rerun()
    
    st.divider()
    
    # Session info
    if st.session_state.session_id:
        try:
            session_info = st.session_state.db.get_current_session(st.session_state.session_id)
            if session_info:
                duration = session_info.get('duration_minutes', 0)
                st.caption(f"Session active for {duration} minutes | User: {st.session_state.user_name}")
        except:
            pass
    
    # Live status badge
    st.markdown('<span class="live-badge">LIVE</span> Real-time monitoring', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Live Monitor", "Analytics"])
    
    # TAB 1: LIVE MONITOR (CLOUD-SAFE)
    with tab1:
        st.markdown("### Real-Time Ergonomic Monitoring")
        
        # Check if we have comparator
        if st.session_state.comparator is None:
            st.warning("System initializing. Please wait...")
            return
        
        # Try WebRTC with comprehensive error handling
        webrtc_placeholder = st.empty()
        
        try:
            # Cache session state values before WebRTC
            comparator = st.session_state.comparator
            user_id = st.session_state.user_id
            session_id = st.session_state.session_id
            db = st.session_state.db
            
            with webrtc_placeholder.container():
                webrtc_ctx = webrtc_streamer(
                    key="visionmate-webrtc-safe",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=lambda: SafeVideoProcessor(comparator, user_id, session_id, db),
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {"ideal": 15}  # Reduced for stability
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
                
                if webrtc_ctx.state.playing:
                    st.success("Camera active - Real-time analysis running")
                    
                    # Show live status cards
                    if st.session_state.last_analysis:
                        results = st.session_state.last_analysis.get('results', {})
                        health = st.session_state.last_analysis.get('health_score', 50)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<div class="status-card">', unsafe_allow_html=True)
                            status_class = "status-normal" if health >= 60 else "status-danger"
                            st.markdown(f'<div class=" {status_class}"> {health:.0f}% </div>', unsafe_allow_html=True)
                            st.markdown('<div style="color:rgba(255,255,255,0.7);">Overall Health</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            eye_result = results.get('eye', {}).get('C1', {})
                            eye_status = "STRAINED" if eye_result.get('classification') == 'STRAINED' else "NORMAL"
                            st.markdown('<div class="status-card">', unsafe_allow_html=True)
                            status_class = "status-danger" if eye_status == "STRAINED" else "status-normal"
                            st.markdown(f'<div class="{status_class}">Eye</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            posture_result = results.get('posture', {}).get('C2', {})
                            posture_status = "SLOUCHING" if posture_result.get('status') == 'SLOUCHING' else "GOOD"
                            st.markdown('<div class="status-card">', unsafe_allow_html=True)
                            status_class = "status-danger" if posture_status == "SLOUCHING" else "status-normal"
                            st.markdown(f'<div class="{status_class}">Posture</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.info("Health score and alerts shown on video + cards above")
                else:
                    st.info("Click START to begin live monitoring")
                    
        except Exception as e:
            st.warning("WebRTC unavailable. Using fallback camera analysis:")
            
            # Fallback camera input
            fallback_img = st.camera_input("Take photo for instant analysis")
            if fallback_img is not None and st.session_state.comparator:
                with st.spinner("Analyzing..."):
                    try:
                        bytes_data = fallback_img.getvalue()
                        nparr = np.frombuffer(bytes_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        results = st.session_state.comparator.process_frame(frame)
                        
                        st.session_state.last_analysis = {
                            'results': results,
                            'health_score': results.get('health_score', 50)
                        }
                        
                        # Show results
                        health = results.get('health_score', 50)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Health Score", f"{health:.0f}%")
                        with col2:
                            eye_status = results['eye'].get('C1', {}).get('classification', 'NORMAL')
                            st.metric("Eye Status", eye_status)
                        with col3:
                            posture_status = results['posture'].get('C2', {}).get('status', 'GOOD')
                            st.metric("Posture", posture_status)
                            
                        st.success("Analysis complete!")
                        
                    except Exception:
                        st.error("Analysis failed")
    
    # TAB 2: ANALYTICS
    with tab2:
        st.markdown("### My Ergonomic Analytics")
        hours = st.selectbox("Time Range", [1, 6, 12, 24, 48, 72], index=3, format_func=lambda x: f"Last {x} hours")
        
        try:
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
                st.markdown("#### Eye Strain Trend")
                fig_eye = px.line(df, x='timestamp', y='eye_score',
                                title='Eye Fatigue Score Over Time',
                                labels={'eye_score': 'Fatigue Score (0=Alert, 1=Strained)', 'timestamp': 'Time'})
                fig_eye.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Strain Threshold")
                fig_eye.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=400)
                st.plotly_chart(fig_eye, use_container_width=True)
                
                st.markdown("#### Posture Trend")
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
                    st.warning("Your average eye strain level is high. Consider more frequent breaks.")
                elif avg_eye > 0.4:
                    st.info("Your average eye strain level is moderate. Monitor your screen time.")
                else:
                    st.success("Your eye health is good.")
                
                if avg_posture > 0.6:
                    st.warning("You slouch frequently. Adjust your chair and monitor position.")
                elif avg_posture > 0.4:
                    st.info("Your posture shows occasional slouching. Check posture hourly.")
                else:
                    st.success("Your posture is excellent!")
                
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
                st.info("Not enough data yet. Continue using VisionMate.")
                
        except Exception as e:
            st.error("Analytics unavailable")

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
