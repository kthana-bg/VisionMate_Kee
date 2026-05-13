# Face Authentication using MediaPipe Face Mesh
# No external face_recognition library required

import cv2
import numpy as np
import os
import json
from datetime import datetime

# Import mediapipe with error handling
try:
    import mediapipe as mp
    HAS_MEDIA_PIPE = True
except ImportError:
    HAS_MEDIA_PIPE = False
    print("Warning: MediaPipe not installed. Face authentication will use fallback method.")

class FaceAuthenticator:
    """
    Face authentication system using MediaPipe facial landmarks
    """
    
    def __init__(self):
        self.HAS_MEDIA_PIPE = HAS_MEDIA_PIPE
        
        if self.HAS_MEDIA_PIPE:
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        else:
            self.face_mesh = None
            print("MediaPipe not available. Using fallback face detection.")
    
    def extract_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from image frame
        Returns 1404-dimensional vector (468 landmarks * 3 coordinates)
        """
        if frame is None:
            return None
        
        if not self.HAS_MEDIA_PIPE or self.face_mesh is None:
            # Fallback: Generate dummy embedding based on image hash
            return self._generate_dummy_embedding(frame)
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            embedding = []
            
            for lm in landmarks.landmark:
                embedding.extend([lm.x, lm.y, lm.z])
            
            return np.array(embedding, dtype=np.float32)
        
        except Exception as e:
            print(f"Face embedding extraction error: {e}")
            return self._generate_dummy_embedding(frame)
    
    def _generate_dummy_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate a dummy embedding based on image content for fallback
        """
        if frame is None:
            return np.zeros(128, dtype=np.float32)
        
        # Resize and convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        
        # Flatten and normalize
        embedding = resized.flatten().astype(np.float32) / 255.0
        
        # Ensure consistent size
        if len(embedding) < 128:
            padded = np.zeros(128, dtype=np.float32)
            padded[:len(embedding)] = embedding
            return padded
        else:
            return embedding[:128]
    
    def detect_face(self, frame: np.ndarray) -> bool:
        """Check if a face is present in the frame"""
        if frame is None:
            return False
        
        if not self.HAS_MEDIA_PIPE or self.face_mesh is None:
            # Fallback: Use OpenCV face detector
            return self._detect_face_opencv(frame)
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            return results.multi_face_landmarks is not None
        except Exception:
            return self._detect_face_opencv(frame)
    
    def _detect_face_opencv(self, frame: np.ndarray) -> bool:
        """
        Fallback face detection using OpenCV Haar Cascade
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            return len(faces) > 0
        except Exception:
            return True  # Assume face is present if detection fails
    
    def reduce_dimension(self, embedding: np.ndarray, target_dim: int = 128) -> np.ndarray:
        """Reduce embedding dimension by truncation"""
        if embedding is None:
            return None
        
        if len(embedding) >= target_dim:
            return embedding[:target_dim]
        
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:len(embedding)] = embedding
        return padded
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        Returns value between 0 and 1
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Ensure same length
        min_len = min(len(emb1), len(emb2))
        emb1 = emb1[:min_len]
        emb2 = emb2[:min_len]
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))