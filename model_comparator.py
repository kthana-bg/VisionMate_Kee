import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque
import tensorflow as tf
import os
import threading
import warnings

# Force CPU for TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Configure TensorFlow for minimal latency
tf.config.set_visible_devices([], 'GPU')
physical_devices = tf.config.list_physical_devices('CPU')
tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.LogicalDeviceConfiguration()]
)

import sys
from io import StringIO

class SuppressEGL:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    def __exit__(self, *args):
        sys.stderr = self._stderr

class ModelComparator:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize custom models
        self.custom_eye_model = None
        self.custom_posture_model = None
        self._load_custom_models()
        
        # Model A1 specific variables
        self.blink_history = deque(maxlen=60)
        self.ear_threshold = 0.25
        self.closed_frames = 0
        
        # Model C2 specific variables
        self.posture_history = deque(maxlen=20)
        self.feature_names = ['neck_angle', 'spine_angle', 'head_tilt', 'shoulder_symmetry',
                              'ear_shoulder_ratio', 'hip_angle', 'thoracic_curve', 'pelvic_tilt']
    
    def _load_custom_models(self):
        # Load trained C1 and C2 models
        
        # Load C1 Eye Model
        c1_path = self.config.get('models', {}).get('eye', {}).get('c1_model_path', 'models/custom_eye_model.keras')
        if os.path.exists(c1_path):
            try:
                self.custom_eye_model = tf.keras.models.load_model(c1_path, compile=False)
                print(f"Loaded C1 model from {c1_path}")
            except Exception as e:
                print(f"Error loading C1 model: {e}")
                self.custom_eye_model = None
        else:
            print(f"C1 model not found at {c1_path}")
            self.custom_eye_model = None
        
        # Load C2 Posture Model
        c2_path = self.config.get('models', {}).get('posture', {}).get('c2_model_path', 'models/custom_posture_model.keras')
        if os.path.exists(c2_path):
            try:
                self.custom_posture_model = tf.keras.models.load_model(c2_path, compile=False)
                print(f"Loaded C2 model from {c2_path}")
            except Exception as e:
                print(f"Error loading C2 model: {e}")
                self.custom_posture_model = None
        else:
            print(f"C2 model not found at {c2_path}")
            self.custom_posture_model = None
    
    def _calculate_ear(self, landmarks, eye_indices):
        # Calculate Eye Aspect Ratio
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
        return ear
    
    def _model_a1(self, face_landmarks, frame):
        # Threshold-based EAR with blink rate analysis
        start = time.perf_counter()
        
        left_idx = [33, 157, 153, 133, 145, 158]
        right_idx = [362, 385, 380, 373, 374, 386]
        
        left_ear = self._calculate_ear(face_landmarks.landmark, left_idx)
        right_ear = self._calculate_ear(face_landmarks.landmark, right_idx)
        avg_ear = (left_ear + right_ear) / 2
        
        # Blink detection
        if avg_ear < self.ear_threshold:
            self.closed_frames += 1
        else:
            if self.closed_frames >= 2:
                self.blink_history.append(time.time())
            self.closed_frames = 0
        
        # Calculate blink rate
        now = time.time()
        recent = [t for t in self.blink_history if now - t <= 60]
        blink_rate = len(recent)
        
        # Determine strain
        if blink_rate < 6:
            score = min(1.0, (6 - blink_rate) / 6)
            classification = 'STRAINED'
        elif blink_rate > 25:
            score = min(1.0, (blink_rate - 25) / 10)
            classification = 'STRAINED'
        else:
            score = 0.2
            classification = 'NORMAL'
        
        latency = (time.perf_counter() - start) * 1000
        
        return {
            'fatigue_score': score,
            'classification': classification,
            'latency_ms': latency,
            'blink_rate': blink_rate,
            'ear': avg_ear
        }
    
    def _model_b1(self, face_landmarks, frame):
        # Simulated MobileNetV2 baseline
        start = time.perf_counter()
        
        import random
        score = random.uniform(0.2, 0.7)
        classification = 'STRAINED' if score > 0.6 else 'NORMAL'
        
        latency = (time.perf_counter() - start) * 1000
        
        return {
            'fatigue_score': score,
            'classification': classification,
            'latency_ms': latency
        }
    
    def _model_c1(self, face_landmarks, frame):
        # Custom trained CNN model - NON-BLOCKING
        start = time.perf_counter()
        
        if self.custom_eye_model is None:
            return {'fatigue_score': 0.5, 'classification': 'NORMAL', 'latency_ms': 0}
        
        h, w = frame.shape[:2]
        
        # Extract eye region
        left_idx = [33, 157, 153, 133, 145, 158]
        points = []
        for idx in left_idx:
            lm = face_landmarks.landmark[idx]
            points.append([int(lm.x * w), int(lm.y * h)])
        
        points = np.array(points)
        x_min = max(0, int(points[:, 0].min()) - 20)
        x_max = min(w, int(points[:, 0].max()) + 20)
        y_min = max(0, int(points[:, 1].min()) - 20)
        y_max = min(h, int(points[:, 1].max()) + 20)
        
        eye_crop = frame[y_min:y_max, x_min:x_max]
        
        if eye_crop.size == 0:
            return {'fatigue_score': 0.5, 'classification': 'NORMAL', 'latency_ms': 0}
        
        # Preprocess
        eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        eye_resized = cv2.resize(eye_gray, (64, 64))
        eye_normalized = eye_resized / 255.0
        input_tensor = np.expand_dims(eye_normalized, axis=(0, -1))
        
        # Use predict_on_batch for single-sample inference (faster than predict)
        pred = self.custom_eye_model.predict_on_batch(input_tensor)[0][0]
        score = float(pred)
        classification = 'STRAINED' if score > 0.5 else 'NORMAL'
        
        latency = (time.perf_counter() - start) * 1000
        
        return {
            'fatigue_score': score,
            'classification': classification,
            'latency_ms': latency
        }
    
    def _extract_posture_features(self, pose_landmarks, frame):
        # Extract 8 features for posture analysis
        h, w = frame.shape[:2]
        lm = pose_landmarks.landmark
        
        features = np.zeros(8, dtype=np.float32)
        
        left_shoulder = np.array([lm[11].x * w, lm[11].y * h])
        right_shoulder = np.array([lm[12].x * w, lm[12].y * h])
        left_hip = np.array([lm[23].x * w, lm[23].y * h])
        right_hip = np.array([lm[24].x * w, lm[24].y * h])
        nose = np.array([lm[0].x * w, lm[0].y * h])
        
        ear_center = (left_shoulder + right_shoulder) / 2
        ear_center[1] = ear_center[1] - 0.15 * h
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # Neck angle
        delta = ear_center - shoulder_center
        features[0] = (np.arctan2(delta[1], delta[0]) * 180 / 180 + 90) / 180
        
        # Spine angle
        delta = shoulder_center - hip_center
        features[1] = (np.arctan2(delta[1], delta[0]) * 180 / 180 + 90) / 180
        
        # Head tilt
        delta = nose - ear_center
        features[2] = (np.arctan2(delta[1], delta[0]) * 180 / 90 + 45) / 90
        
        # Shoulder symmetry
        features[3] = abs(left_shoulder[1] - right_shoulder[1]) / h
        
        # Ear-shoulder ratio
        ear_to_shoulder = np.linalg.norm(ear_center - shoulder_center)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
        features[4] = ear_to_shoulder / shoulder_width
        
        # Hip angle
        hip_vec = shoulder_center - hip_center
        vertical = np.array([0, 1])
        dot = np.dot(hip_vec, vertical)
        norm = np.linalg.norm(hip_vec) + 1e-6
        features[5] = np.arccos(np.clip(dot / norm, -1, 1)) / np.pi
        
        # Thoracic curve
        mid_spine = (shoulder_center + hip_center) / 2
        curve = mid_spine - shoulder_center
        dot = np.dot(curve, vertical)
        norm = np.linalg.norm(curve) + 1e-6
        features[6] = np.arccos(np.clip(dot / norm, -1, 1)) / np.pi
        
        # Pelvic tilt
        dot = np.dot(hip_vec, vertical)
        norm = np.linalg.norm(hip_vec) + 1e-6
        features[7] = np.arccos(np.clip(dot / norm, -1, 1)) / np.pi
        
        return np.clip(features, 0, 1)
    
    def _model_a2(self, pose_landmarks, frame):
        # Angle-based posture detection
        start = time.perf_counter()
        
        h, w = frame.shape[:2]
        lm = pose_landmarks.landmark
        
        left_shoulder = np.array([lm[11].x * w, lm[11].y * h])
        right_shoulder = np.array([lm[12].x * w, lm[12].y * h])
        left_ear = np.array([lm[7].x * w, lm[7].y * h])
        right_ear = np.array([lm[8].x * w, lm[8].y * h])
        
        ear_center = (left_ear + right_ear) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        delta = ear_center - shoulder_center
        angle = abs(np.arctan2(delta[1], delta[0]) * 180 / 180)
        
        if 70 <= angle <= 110:
            score = 0.2
            status = 'GOOD'
        else:
            score = min(1.0, abs(angle - 90) / 45)
            status = 'SLOUCHING'
        
        latency = (time.perf_counter() - start) * 1000
        
        return {
            'posture_score': score,
            'status': status,
            'latency_ms': latency,
            'angle': angle
        }
    
    def _model_b2(self, pose_landmarks, frame):
        # Simulated BlazePose baseline
        start = time.perf_counter()
        
        import random
        score = random.uniform(0.2, 0.7)
        status = 'SLOUCHING' if score > 0.6 else 'GOOD'
        
        latency = (time.perf_counter() - start) * 1000
        
        return {
            'posture_score': score,
            'status': status,
            'latency_ms': latency
        }
    
    def _model_c2(self, pose_landmarks, frame):
        # Custom trained LSTM model - NON-BLOCKING
        start = time.perf_counter()
        
        if self.custom_posture_model is None:
            return {'slouching_prob': 0.5, 'status': 'GOOD', 'latency_ms': 0}
        
        features = self._extract_posture_features(pose_landmarks, frame)
        self.posture_history.append(features)
        
        if len(self.posture_history) < 20:
            return {'slouching_prob': 0.5, 'status': 'LEARNING', 'latency_ms': 0}
        
        sequence = np.array(self.posture_history)
        input_tensor = np.expand_dims(sequence, axis=0)
        
        # Use predict_on_batch for single-sample inference (faster than predict)
        pred = self.custom_posture_model.predict_on_batch(input_tensor)[0][0]
        prob = float(pred)
        status = 'SLOUCHING' if prob > 0.6 else 'GOOD'
        
        latency = (time.perf_counter() - start) * 1000
        
        return {
            'slouching_prob': prob,
            'status': status,
            'latency_ms': latency
        }
    
    def process_frame(self, frame: np.ndarray) -> dict:
        # Process a single frame with all 6 models
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)
        
        eye_results = {}
        posture_results = {}
        
        # Eye models
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0]
            eye_results['A1'] = self._model_a1(landmarks, frame)
            eye_results['B1'] = self._model_b1(landmarks, frame)
            eye_results['C1'] = self._model_c1(landmarks, frame)
        else:
            for m in ['A1', 'B1', 'C1']:
                eye_results[m] = {'fatigue_score': 0.5, 'classification': 'NO_FACE', 'latency_ms': 0}
        
        # Posture models
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks
            posture_results['A2'] = self._model_a2(landmarks, frame)
            posture_results['B2'] = self._model_b2(landmarks, frame)
            posture_results['C2'] = self._model_c2(landmarks, frame)
        else:
            for m in ['A2', 'B2', 'C2']:
                posture_results[m] = {'posture_score': 0.5, 'status': 'NO_POSE', 'latency_ms': 0}
        
        # Calculate consensus
        eye_strained = sum(1 for m in eye_results.values() if m.get('classification') == 'STRAINED')
        posture_poor = sum(1 for m in posture_results.values() if m.get('status') == 'SLOUCHING')
        
        eye_consensus = 'STRAINED' if eye_strained >= 2 else 'NORMAL'
        posture_consensus = 'SLOUCHING' if posture_poor >= 2 else 'GOOD'
        
        # Calculate health score
        c1_score = eye_results.get('C1', {}).get('fatigue_score', 0.5)
        c2_score = posture_results.get('C2', {}).get('slouching_prob', 0.5)
        health_score = int(100 - (c1_score * 40) - (c2_score * 40))
        health_score = max(0, min(100, health_score))
        
        return {
            'eye': eye_results,
            'posture': posture_results,
            'eye_consensus': eye_consensus,
            'posture_consensus': posture_consensus,
            'eye_strained_count': eye_strained,
            'posture_poor_count': posture_poor,
            'health_score': health_score
        }
