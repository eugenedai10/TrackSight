"""
Camera utilities for gaze estimation system.

Provides a unified interface for camera management across different modules.
"""

import cv2
import logging


class CameraManager:
    """
    Manages camera capture with consistent configuration.
    
    Provides initialization, frame reading, and cleanup functionality
    for OpenCV camera capture.
    """
    
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        """
        Initialize camera manager.
        
        Args:
            camera_index: Camera device index (default: 0)
            width: Frame width in pixels (default: 640)
            height: Frame height in pixels (default: 480)
            fps: Frames per second (default: 30)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def initialize(self):
        """
        Initialize and configure the camera.
        
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            IOError: If camera cannot be opened
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        logging.info(f"Camera {self.camera_index} initialized: {self.width}x{self.height} @ {self.fps}fps")
        return True
    
    def read_frame(self):
        """
        Read a frame from the camera.
        
        Returns:
            tuple: (success, frame) where success is bool and frame is numpy array
        """
        if self.cap is None:
            logging.warning("Camera not initialized, cannot read frame")
            return False, None
        
        return self.cap.read()
    
    def is_opened(self):
        """
        Check if camera is currently opened.
        
        Returns:
            bool: True if camera is opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logging.info(f"Camera {self.camera_index} released")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
