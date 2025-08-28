import streamlit as st
import requests
import json
import pandas as pd
import time
from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import io

st.set_page_config(
    page_title="Video Text Detection",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None
    
    def login(self, username: str, password: str) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/auth/login",
                data={"username": username, "password": password}
            )
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                return True
            return False
        except Exception as e:
            st.error(f"Login failed: {e}")
            return False
    
    def register(self, email: str, username: str, password: str) -> bool:
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/auth/register",
                json={"email": email, "username": username, "password": password}
            )
            if response.status_code == 201:
                self.token = response.json()["access_token"]
                return True
            return False
        except Exception as e:
            st.error(f"Registration failed: {e}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}
    
    def upload_video(self, file, category: Optional[str] = None) -> Optional[Dict]:
        try:
            files = {"file": file}
            data = {"category": category} if category else {}
            
            response = requests.post(
                f"{self.base_url}/api/v1/videos/upload",
                files=files,
                data=data,
                headers=self.get_headers()
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                st.error(f"Upload failed: {response.text}")
                return None
        except Exception as e:
            st.error(f"Upload error: {e}")
            return None
    
    def get_videos(self) -> List[Dict]:
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/videos/",
                headers=self.get_headers()
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Failed to get videos: {e}")
            return []
    
    def start_processing(self, video_id: int, confidence: float = 0.5, use_transformer: bool = True) -> Optional[Dict]:
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/processing/videos/{video_id}/detect",
                params={
                    "confidence_threshold": confidence,
                    "use_transformer": use_transformer
                },
                headers=self.get_headers()
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to start processing: {e}")
            return None
    
    def get_job_status(self, job_id: int) -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/processing/jobs/{job_id}/status",
                headers=self.get_headers()
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to get job status: {e}")
            return None
    
    def get_results(self, video_id: int, format: str = "json") -> Optional[Dict]:
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/processing/videos/{video_id}/results",
                params={"format": format},
                headers=self.get_headers()
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to get results: {e}")
            return None

def init_session_state():
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient(API_BASE_URL)
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None

def login_page():
    st.title("🔐 Login to Video Text Detection")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                if st.session_state.api_client.login(username, password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            email = st.text_input("Email")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Register"):
                if st.session_state.api_client.register(email, username, password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.success("Registration successful!")
                    st.rerun()

def main_app():
    st.title("📹 Video Text Detection System")
    
    with st.sidebar:
        st.write(f"👤 Welcome, {st.session_state.current_user}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.api_client.token = None
            st.rerun()
        
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["📤 Upload Video", "📋 My Videos", "⚙️ Processing", "📊 Results", "📈 Analytics"]
        )
    
    if page == "📤 Upload Video":
        upload_page()
    elif page == "📋 My Videos":
        videos_page()
    elif page == "⚙️ Processing":
        processing_page()
    elif page == "📊 Results":
        results_page()
    elif page == "📈 Analytics":
        analytics_page()

def upload_page():
    st.header("Upload Video for Text Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV. Max size: 500MB"
        )
        
        if uploaded_file:
            st.video(uploaded_file)
            
            category = st.selectbox(
                "Video Category (Optional)",
                ["", "activity", "driving", "game", "sports", "street_indoor", "street_outdoor", "other"]
            )
            
            if st.button("Upload Video", type="primary"):
                with st.spinner("Uploading video..."):
                    result = st.session_state.api_client.upload_video(
                        uploaded_file,
                        category if category else None
                    )
                    
                    if result:
                        st.success(f"Video uploaded successfully! ID: {result['id']}")
                        st.json(result)
    
    with col2:
        st.info("📝 **Upload Guidelines**")
        st.write("• Max file size: 500MB")
        st.write("• Max duration: 5 minutes") 
        st.write("• Best quality: HD (720p+)")
        st.write("• Clear text preferred")

def videos_page():
    st.header("My Videos")
    
    videos = st.session_state.api_client.get_videos()
    
    if videos:
        for video in videos:
            with st.expander(f"📹 {video['original_filename']} (ID: {video['id']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Size:** {video['file_size'] / 1024 / 1024:.1f} MB")
                    st.write(f"**Category:** {video.get('category', 'N/A')}")
                
                with col2:
                    if video.get('duration'):
                        st.write(f"**Duration:** {video['duration']:.1f}s")
                    if video.get('fps'):
                        st.write(f"**FPS:** {video['fps']:.1f}")
                
                with col3:
                    st.write(f"**Uploaded:** {video['created_at'][:10]}")
                    if video.get('width') and video.get('height'):
                        st.write(f"**Resolution:** {video['width']}x{video['height']}")
    else:
        st.info("No videos uploaded yet. Go to Upload Video to get started!")

def processing_page():
    st.header("Video Processing")
    
    videos = st.session_state.api_client.get_videos()
    
    if videos:
        video_options = {f"{v['original_filename']} (ID: {v['id']})": v['id'] for v in videos}
        selected_video = st.selectbox("Select Video", list(video_options.keys()))
        
        if selected_video:
            video_id = video_options[selected_video]
            
            col1, col2 = st.columns(2)
            
            with col1:
                confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
                use_transformer = st.checkbox("Use Transformer OCR", value=True, help="More accurate but slower")
            
            with col2:
                st.info("⚙️ **Processing Settings**")
                st.write("• Higher confidence = fewer false positives")
                st.write("• Transformer OCR = better accuracy")
                st.write("• Processing time: 1-5 minutes per minute of video")
            
            if st.button("Start Processing", type="primary"):
                with st.spinner("Starting processing..."):
                    job = st.session_state.api_client.start_processing(
                        video_id, confidence, use_transformer
                    )
                    
                    if job:
                        st.success(f"Processing started! Job ID: {job['id']}")
                        st.session_state.current_job = job['id']
                        
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()
                        
                        while True:
                            status = st.session_state.api_client.get_job_status(job['id'])
                            
                            if status:
                                progress_placeholder.progress(status['progress'] / 100)
                                status_placeholder.write(f"Status: {status['status']} | Progress: {status['progress']:.1f}%")
                                
                                if status['status'] in ['completed', 'failed', 'cancelled']:
                                    break
                            
                            time.sleep(2)
                        
                        if status['status'] == 'completed':
                            st.success("Processing completed successfully!")
                        else:
                            st.error(f"Processing {status['status']}: {status.get('error_message', '')}")

def results_page():
    st.header("Processing Results")
    
    videos = st.session_state.api_client.get_videos()
    
    if videos:
        video_options = {f"{v['original_filename']} (ID: {v['id']})": v['id'] for v in videos}
        selected_video = st.selectbox("Select Video", list(video_options.keys()))
        
        if selected_video:
            video_id = video_options[selected_video]
            
            results = st.session_state.api_client.get_results(video_id)
            
            if results and results.get('results'):
                st.success("✅ Results found!")
                
                summary = results.get('summary', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", summary.get('total_frames', 0))
                with col2:
                    st.metric("Frames with Text", summary.get('frames_with_text', 0))
                with col3:
                    st.metric("Total Detections", summary.get('total_detections', 0))
                with col4:
                    st.metric("Unique Texts", summary.get('unique_texts', 0))
                
                st.subheader("📊 Detection Summary")
                
                if summary.get('detected_texts'):
                    st.write("**Detected Texts:**")
                    for i, text in enumerate(summary['detected_texts'][:20], 1):
                        st.write(f"{i}. {text}")
                    
                    if len(summary['detected_texts']) > 20:
                        st.write(f"... and {len(summary['detected_texts']) - 20} more")
                
                st.subheader("📋 Detailed Results")
                
                tab1, tab2, tab3 = st.tabs(["Table View", "JSON View", "Export"])
                
                with tab1:
                    detections_data = []
                    for frame in results['results']['results']:
                        for det in frame['detections']:
                            detections_data.append({
                                'Frame': frame['frame_number'],
                                'Timestamp': f"{frame['timestamp']:.2f}s",
                                'Text': det['text'],
                                'Confidence': f"{det['detection_confidence']:.3f}",
                                'X1': det['bbox'][0],
                                'Y1': det['bbox'][1],
                                'X2': det['bbox'][2], 
                                'Y2': det['bbox'][3]
                            })
                    
                    if detections_data:
                        df = pd.DataFrame(detections_data)
                        st.dataframe(df, use_container_width=True)
                
                with tab2:
                    st.json(results)
                
                with tab3:
                    csv_results = st.session_state.api_client.get_results(video_id, "csv")
                    if csv_results:
                        st.download_button(
                            "Download CSV",
                            csv_results['content'],
                            f"video_{video_id}_results.csv",
                            "text/csv"
                        )
            else:
                st.info("No results found. Please process the video first.")

def analytics_page():
    st.header("Analytics Dashboard")
    
    videos = st.session_state.api_client.get_videos()
    
    if videos:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Video Categories")
            categories = [v.get('category', 'unknown') for v in videos]
            category_counts = pd.Series(categories).value_counts()
            
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribution by Category"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📈 Upload Timeline")
            upload_dates = [v['created_at'][:10] for v in videos]
            date_counts = pd.Series(upload_dates).value_counts().sort_index()
            
            fig_line = px.line(
                x=date_counts.index,
                y=date_counts.values,
                title="Videos Uploaded Over Time"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        st.subheader("📋 Video Statistics")
        video_df = pd.DataFrame(videos)
        
        if 'file_size' in video_df.columns:
            video_df['file_size_mb'] = video_df['file_size'] / 1024 / 1024
        
        st.dataframe(
            video_df[['original_filename', 'category', 'file_size_mb', 'duration', 'created_at']].fillna('N/A'),
            use_container_width=True
        )

def main():
    init_session_state()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()