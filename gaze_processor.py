#!/usr/bin/env python3
"""
Gaze Processor Module

Handles real-time gaze estimation using inference components.
Shared by multiple applications for consistent gaze estimation.
"""

import cv2
import time
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms
from config import data_config
from utils.helpers import get_model, GazeSmoother
import uniface
import onnxruntime as ort


class GazeProcessor:
    """Handles real-time gaze estimation using inference components."""
    
    def __init__(self, model="resnet34", weight="weights/resnet34.pt", dataset="gaze360"):
        self.model_name = model
        self.weight_path = weight
        self.dataset = dataset
        
        # Get dataset configuration
        if dataset in data_config:
            dataset_config = data_config[dataset]
            self.bins = dataset_config["bins"]
            self.binwidth = dataset_config["binwidth"]
            self.angle = dataset_config["angle"]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        self.device = self._setup_device()
        self.idx_tensor = torch.arange(self.bins, device=self.device, dtype=torch.float32)
        
        # Monkey-patch uniface to enable CoreML acceleration for face detection on Apple Silicon
        self._enable_coreml_face_detection()
        
        # Initialize models
        self.face_detector = uniface.RetinaFace("retinaface_mnet025")
        self.gaze_detector = self._load_gaze_model()
        
        # Smoothing
        self.gaze_smoother = GazeSmoother(0.1)  # 100ms smoothing
        
        # Current gaze data
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.last_update_time = 0.0
        self.is_face_detected = False
        
        # Face detection optimization: run every N frames
        self.frame_count = 0
        self.face_detection_interval = 3
        self.last_bbox = None
        self.last_keypoints = None
    
    def _setup_device(self):
        """Setup computation device."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using Apple M2 GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU")
        return device
    
    def _enable_coreml_face_detection(self):
        """Enable CoreML acceleration for face detection on Apple Silicon."""
        original_init_model = uniface.RetinaFace._initialize_model
        
        def patched_init_model(self, model_path: str) -> None:
            """Modified initialization to use CoreML execution provider for Apple Silicon."""
            try:
                # Try CoreML first (uses Apple GPU and Neural Engine)
                available_providers = ort.get_available_providers()
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if 'CoreMLExecutionProvider' in available_providers else ['CPUExecutionProvider']
                
                self.session = ort.InferenceSession(model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                
                used_provider = self.session.get_providers()[0]
                if used_provider == 'CoreMLExecutionProvider':
                    logging.info(f"Face detector using CoreML (Apple GPU/Neural Engine)")
                else:
                    logging.info(f"Face detector using {used_provider}")
            except Exception as e:
                logging.error(f"Failed to load model from '{model_path}': {e}")
                raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e
        
        # Apply the patch
        uniface.RetinaFace._initialize_model = patched_init_model
    
    def _load_gaze_model(self):
        """Load gaze estimation model."""
        try:
            gaze_detector = get_model(self.model_name, self.bins, inference_mode=True)
            state_dict = torch.load(self.weight_path, map_location=self.device, weights_only=False)
            gaze_detector.load_state_dict(state_dict)
            gaze_detector.to(self.device)
            gaze_detector.eval()
            logging.info("Gaze estimation model loaded successfully")
            return gaze_detector
        except Exception as e:
            logging.error(f"Failed to load gaze model: {e}")
            raise
    
    def _preprocess_image(self, image):
        """Preprocess image for gaze estimation."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        return image.unsqueeze(0)
    
    def process_frame(self, frame):
        """Process a single frame and update gaze data."""
        current_time = time.time()
        self.is_face_detected = False
        
        try:
            # Run face detection every N frames
            if self.frame_count % self.face_detection_interval == 0:
                bboxes, keypoints = self.face_detector.detect(frame)
                
                if len(bboxes) > 0:
                    # Cache the first detected face
                    self.last_bbox = bboxes[0]
                    self.last_keypoints = keypoints[0] if len(keypoints) > 0 else None
                else:
                    # No face detected, clear cache
                    self.last_bbox = None
                    self.last_keypoints = None
            
            # Increment frame counter
            self.frame_count += 1
            
            # Use cached bbox for gaze estimation (if available)
            if self.last_bbox is not None:
                bbox = self.last_bbox
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                
                # Extract face region
                face_image = frame[y_min:y_max, x_min:x_max]
                if face_image.size > 0:
                    # Preprocess and run gaze estimation
                    image_tensor = self._preprocess_image(face_image)
                    image_tensor = image_tensor.to(self.device)
                    
                    with torch.no_grad():
                        pitch, yaw = self.gaze_detector(image_tensor)
                        
                        pitch_predicted = F.softmax(pitch, dim=1)
                        yaw_predicted = F.softmax(yaw, dim=1)
                        
                        # Convert to angles
                        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, dim=1) * self.binwidth - self.angle
                        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, dim=1) * self.binwidth - self.angle
                        
                        # Convert to degrees
                        pitch_deg = float(pitch_predicted.cpu())
                        yaw_deg = float(yaw_predicted.cpu())
                        
                        # Apply smoothing
                        pitch_smooth, yaw_smooth = self.gaze_smoother.update(pitch_deg, yaw_deg, current_time)
                        
                        self.current_pitch = pitch_smooth
                        self.current_yaw = yaw_smooth
                        self.last_update_time = current_time
                        self.is_face_detected = True
                        
        except Exception as e:
            logging.warning(f"Frame processing failed: {e}")
    
    def get_current_gaze(self):
        """Get current gaze angles."""
        return self.current_pitch, self.current_yaw, self.is_face_detected
