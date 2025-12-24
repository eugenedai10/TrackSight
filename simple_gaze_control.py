#!/usr/bin/env python3
"""
Simple Gaze Control Application

Uses calibration-based linear regression models to map gaze angles to screen coordinates.
Requires calibration data from calibration_simple.py
"""

import cv2
import numpy as np
import time
import logging
import sys
import argparse
import threading
import queue
from collections import deque

try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to stop
    pyautogui.PAUSE = 0.01  # Minimal pause between commands
except ImportError:
    print("PyAutoGUI not installed. Run: pip install pyautogui")
    sys.exit(1)

try:
    from AppKit import NSScreen
except ImportError:
    print("AppKit not available. This script requires macOS with PyObjC.")
    sys.exit(1)

# Import gaze estimation components
from gaze_processor import GazeProcessor
from uart_reader import UARTReader, uart_control_state, uart_button_states
from send_link_command import send_command
from calibration.camera_utils import CameraManager
from calibration.data_utils import load_calibration_data
from calibration.model_utils import CalibrationModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class FPSCounter:
    """Thread-safe FPS counter"""
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def update(self):
        with self.lock:
            self.times.append(time.time())
    
    def get_fps(self):
        with self.lock:
            if len(self.times) < 2:
                return 0.0
            elapsed = self.times[-1] - self.times[0]
            if elapsed == 0:
                return 0.0
            return (len(self.times) - 1) / elapsed


def camera_thread(camera_manager, frame_queue, stop_event, camera_fps):
    """Background thread for camera capture"""
    logging.info("Camera thread started")
    
    while not stop_event.is_set():
        success, frame = camera_manager.read_frame()
        if not success:
            logging.warning("Camera thread: Failed to read frame")
            time.sleep(0.01)
            continue
        
        camera_fps.update()
        
        try:
            # Non-blocking put - drop frame if queue full
            frame_queue.put((time.time(), frame.copy()), timeout=0.01)
        except queue.Full:
            pass  # Drop frame if inference is slow
    
    logging.info("Camera thread stopped")


def gaze_estimation_thread(gaze_processor, gaze_mapper, frame_queue, 
                          result_queue, stop_event, inference_fps):
    """Background thread for gaze estimation"""
    logging.info("Gaze estimation thread started")
    
    while not stop_event.is_set():
        try:
            timestamp, frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Process frame for gaze estimation
        gaze_processor.process_frame(frame)
        pitch, yaw, face_detected = gaze_processor.get_current_gaze()
        
        # Convert to screen coordinates
        screen_x, screen_y, screen_idx = None, None, None
        if face_detected:
            screen_x, screen_y, screen_idx = gaze_mapper.gaze_to_screen(pitch, yaw)
        
        inference_fps.update()
        
        try:
            # Send results to main thread
            result_queue.put((timestamp, frame, pitch, yaw, face_detected, 
                            screen_x, screen_y, screen_idx), timeout=0.01)
        except queue.Full:
            pass  # Drop if display is slow
    
    logging.info("Gaze estimation thread stopped")


class CalibrationGazeMapper:
    """Calibration-based mapping for gaze to screen coordinates."""
    
    def __init__(self, calibration_file='simple_calibration.csv'):
        self.screen_width, self.screen_height = pyautogui.size()
        logging.info(f"Screen size: {self.screen_width}x{self.screen_height}")
        
        # Load calibration data and create model
        try:
            logging.info(f"Using calibration file: {calibration_file}")
            data = load_calibration_data(calibration_file)
            self.calib_model = CalibrationModel(data)
            
            # Log model quality
            for screen_idx in sorted(self.calib_model.model_params.keys()):
                params = self.calib_model.model_params[screen_idx]
                logging.info(f"Screen {screen_idx} - Pitch R²={params['pitch']['r_squared']:.4f}, "
                           f"Yaw R²={params['yaw']['r_squared']:.4f}")
            
            logging.info("Calibration-based mapping initialized successfully")
        except FileNotFoundError:
            logging.error(f"Calibration file '{calibration_file}' not found!")
            logging.error("Please run calibration first using calibration_simple.py")
            raise
        except Exception as e:
            logging.error(f"Failed to load calibration: {e}")
            raise
    
    def gaze_to_screen(self, pitch, yaw):
        """Convert gaze angles to screen coordinates using calibration models.
        
        Returns:
            tuple: (x, y, screen_idx) - absolute screen coordinates and screen index
        """
        if pitch is None or yaw is None:
            return None, None, None
        
        try:
            # Use calibration-based prediction (returns absolute coordinates + screen_index)
            # Note: predict() already clamps to proper screen bounds, no need to clamp again
            x, y, screen_idx = self.calib_model.predict(pitch, yaw)
            
            return int(x), int(y), screen_idx
        except Exception as e:
            logging.warning(f"Prediction failed: {e}")
            return None, None, None


class MouseSmoother:
    """Smooths mouse coordinates using simple moving average of last N points with circular dead zone."""
    
    def __init__(self, window_size=3, min_movement=30):
        self.window_size = window_size  # Number of points to average (N)
        self.min_movement = min_movement  # Dead zone radius in pixels
        self.history_x = deque(maxlen=window_size)
        self.history_y = deque(maxlen=window_size)
        self.last_x = None  # Track last returned position
        self.last_y = None
    
    def update(self, target_x, target_y):
        """Apply simple moving average smoothing to coordinates with circular dead zone.
        
        Returns:
            tuple: (x, y, distance, moved) where moved is True if position changed
        """
        # Add new coordinates to history
        self.history_x.append(float(target_x))
        self.history_y.append(float(target_y))
        
        # Calculate average of all points in history
        smoothed_x = sum(self.history_x) / len(self.history_x)
        smoothed_y = sum(self.history_y) / len(self.history_y)
        
        # First call - initialize last position
        if self.last_x is None:
            self.last_x = smoothed_x
            self.last_y = smoothed_y
            return int(smoothed_x), int(smoothed_y), 0.0, True
        
        # Apply circular dead zone: compare smoothed vs last returned position
        distance = 0.0
        moved = True
        
        if self.min_movement > 0:
            # Calculate Euclidean distance from new smoothed position to last returned position
            distance = np.sqrt((smoothed_x - self.last_x)**2 + (smoothed_y - self.last_y)**2)
            
            if distance < self.min_movement:
                # Within dead zone - return last position (hold still)
                return int(self.last_x), int(self.last_y), distance, False
        
        # Outside dead zone - move to new smoothed position
        self.last_x = smoothed_x
        self.last_y = smoothed_y
        return int(smoothed_x), int(smoothed_y), distance, True


class MouseController:
    """Handles mouse movement with debug mode option and coordinate conversion."""
    
    def __init__(self, debug_mode=True, window_size=3, min_movement=30):
        self.debug_mode = debug_mode
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Get main screen height for coordinate conversion
        # Cocoa calibration uses bottom-left origin, PyAutoGUI uses top-left
        main_screen = NSScreen.mainScreen()
        self.main_screen_height = main_screen.frame().size.height
        
        # Initialize mouse smoother with SMA
        self.mouse_smoother = MouseSmoother(window_size, min_movement)
        
        if not debug_mode:
            self.check_permissions()
        
        logging.info(f"Screen size: {self.screen_width}x{self.screen_height}")
        logging.info(f"Main screen height: {self.main_screen_height:.0f} (for Cocoa→PyAutoGUI conversion)")
        logging.info(f"Mouse control debug mode: {'ON' if debug_mode else 'OFF'}")
        logging.info(f"Mouse smoothing: SMA window_size={window_size}, min movement={min_movement}px")
    
    def check_permissions(self):
        """Check and request accessibility permissions."""
        try:
            # Test if we can get mouse position
            pyautogui.position()
            logging.info("Accessibility permissions OK")
        except Exception as e:
            logging.error("Accessibility permissions required!")
            logging.error("Go to: System Preferences → Security & Privacy → Privacy → Accessibility")
            sys.exit(1)
    
    def move_to(self, cocoa_x, cocoa_y):
        """Move mouse to absolute position (expects Cocoa coordinates, converts to PyAutoGUI).
        
        Args:
            cocoa_x: X coordinate in Cocoa system (same as PyAutoGUI)
            cocoa_y: Y coordinate in Cocoa system (bottom-left origin, Y increases upward)
        """
        # Convert from Cocoa (bottom-left origin) to PyAutoGUI (top-left origin) coordinates
        pyautogui_x = cocoa_x
        pyautogui_y = self.main_screen_height - cocoa_y
        
        # Apply mouse smoothing to converted coordinates
        smoothed_x, smoothed_y, distance, moved = self.mouse_smoother.update(pyautogui_x, pyautogui_y)
        
        # Create status string
        status = "MOVE" if moved else "HOLD"
        
        # Combined log line
        logging.info(f"Mouse: Cocoa=({cocoa_x:.0f}, {cocoa_y:.0f}) → "
                    f"Smoothed=({smoothed_x}, {smoothed_y}) | "
                    f"Δ{distance:.0f}px {status}")
        
        if not self.debug_mode:
            # ACTUAL MODE: Move the mouse
            try:
                pyautogui.moveTo(smoothed_x, smoothed_y)
            except Exception as e:
                logging.warning(f"Mouse move failed: {e}")


class SimpleGazeApp:
    """Main simple gaze control application."""
    
    def __init__(self, camera_index=0, debug_mode=False, timeout_minutes=3, window_size=3, min_movement=30, uart_port="/dev/tty.usbmodem141301", calibration_file='simple_calibration.csv'):
        self.gaze_mapper = CalibrationGazeMapper(calibration_file)
        self.gaze_processor = GazeProcessor()
        self.mouse_controller = MouseController(debug_mode, window_size, min_movement)
        
        # Camera setup
        self.camera_manager = CameraManager(camera_index=camera_index)
        
        # UART setup
        self.uart_reader = UARTReader(port=uart_port, baudrate=115200)
        
        # Control state
        self.running = False
        self.gaze_control_enabled = True
        self.show_preview = True
        
        # Timeout settings
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = time.time()
        
        # Statistics
        self.frame_count = 0
        self.last_fps_time = None
        self.last_frame_count = 0
        
        # Button state tracking for edge detection
        self.previous_button_states = [0, 0, 0]
        
        # Threading components
        self.frame_queue = None
        self.result_queue = None
        self.stop_event = None
        self.camera_fps = None
        self.inference_fps = None
        self.display_fps = None
        self.cam_thread = None
        self.inf_thread = None
        
    def initialize_camera(self):
        """Initialize camera capture."""
        self.camera_manager.initialize()
    
    def process_keyboard_input(self):
        """Handle keyboard input for control."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            self.running = False
        elif key == ord(' '):  # Space to toggle gaze control
            self.gaze_control_enabled = not self.gaze_control_enabled
            status = "ENABLED" if self.gaze_control_enabled else "DISABLED"
            logging.info(f"Gaze control {status}")
        elif key == ord('p'):  # 'p' to toggle preview
            self.show_preview = not self.show_preview
    
    def run(self):
        """Main application loop with parallel processing."""
        try:
            self.initialize_camera()
            self.uart_reader.start()
            self.running = True
            
            # Create queues and FPS counters
            self.frame_queue = queue.Queue(maxsize=2)
            self.result_queue = queue.Queue(maxsize=2)
            self.stop_event = threading.Event()
            self.camera_fps = FPSCounter()
            self.inference_fps = FPSCounter()
            self.display_fps = FPSCounter()
            
            logging.info("=== Simple Gaze Control Started (Parallel Mode) ===")
            logging.info("Pipeline: Camera -> Inference -> Mouse Control + Display")
            logging.info("Controls:")
            logging.info("  SPACE - Toggle gaze control on/off")
            logging.info("  P - Toggle camera preview")
            logging.info("  Q/ESC - Quit application")
            
            # Start background threads
            self.cam_thread = threading.Thread(
                target=camera_thread,
                args=(self.camera_manager, self.frame_queue, self.stop_event, self.camera_fps),
                daemon=True
            )
            
            self.inf_thread = threading.Thread(
                target=gaze_estimation_thread,
                args=(self.gaze_processor, self.gaze_mapper, self.frame_queue,
                      self.result_queue, self.stop_event, self.inference_fps),
                daemon=True
            )
            
            self.cam_thread.start()
            self.inf_thread.start()
            
            # Main loop handles UART, mouse control, and display
            self._main_display_loop()
            
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error(f"Application error: {e}")
        finally:
            self.cleanup()
    
    def _main_display_loop(self):
        """Main thread loop: processes UART, controls mouse, displays preview."""
        
        while self.running and not self.stop_event.is_set():
            # Check timeout
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.timeout_seconds:
                logging.info(f"Timeout reached. Exiting...")
                self.stop_event.set()
                break
            
            # Process UART state
            if uart_control_state[0] == 1:
                self.gaze_control_enabled = True
            else:
                self.gaze_control_enabled = False
            
            # Check button presses
            for i in range(3):
                if uart_button_states[i] == 1 and self.previous_button_states[i] == 0:
                    message = {'command': 'click-link', 'number': i + 1}
                    logging.info(f"Button {i+1} pressed, sending command")
                    send_command(message)
                self.previous_button_states[i] = uart_button_states[i]
            
            # Get gaze estimation results
            try:
                timestamp, frame, pitch, yaw, face_detected, screen_x, screen_y, screen_idx = \
                    self.result_queue.get(timeout=0.1)
            except queue.Empty:
                self.process_keyboard_input()
                continue
            
            self.frame_count += 1
            self.display_fps.update()
            
            # Calculate latency
            latency_ms = (time.time() - timestamp) * 1000
            
            # Move mouse if gaze control enabled
            if face_detected and self.gaze_control_enabled and screen_x is not None:
                self.mouse_controller.move_to(screen_x, screen_y)
            
            # Display preview
            if self.show_preview:
                self.draw_preview_parallel(frame, pitch, yaw, face_detected, 
                                          screen_x, screen_y, screen_idx, latency_ms)
                cv2.imshow('Simple Gaze Control', frame)
            
            # Handle keyboard
            self.process_keyboard_input()
            
            # Log statistics periodically
            if self.frame_count % 150 == 0:
                self._log_parallel_stats(latency_ms)
    
    def draw_preview_parallel(self, frame, pitch, yaw, face_detected, 
                             screen_x, screen_y, screen_idx, latency_ms):
        """Draw preview with parallel pipeline stats."""
        height, width = frame.shape[:2]
        
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        control_status = "ON" if self.gaze_control_enabled else "OFF"
        
        # Status info
        cv2.putText(frame, f"Face: {'Detected' if face_detected else 'Not Found'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Gaze Control: {control_status}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}°  Yaw: {yaw:.1f}°", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Screen coordinates
        if face_detected and screen_x is not None:
            cv2.putText(frame, f"Screen: ({screen_x}, {screen_y}) [Screen {screen_idx}]", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Pipeline stats
        cam_fps = self.camera_fps.get_fps()
        inf_fps = self.inference_fps.get_fps()
        disp_fps = self.display_fps.get_fps()
        
        cv2.putText(frame, f"Cam: {cam_fps:.1f} | Inf: {inf_fps:.1f} | Disp: {disp_fps:.1f} FPS", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.0f}ms", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "SPACE: Toggle  P: Preview  Q: Quit", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _log_parallel_stats(self, latency_ms):
        """Log parallel pipeline statistics."""
        cam_fps = self.camera_fps.get_fps()
        inf_fps = self.inference_fps.get_fps()
        disp_fps = self.display_fps.get_fps()
        
        logging.info(f"Pipeline FPS: Cam={cam_fps:.1f} | Inf={inf_fps:.1f} | Disp={disp_fps:.1f}, "
                    f"Latency={latency_ms:.0f}ms")
    
    def draw_preview(self, frame, pitch, yaw, face_detected):
        """Draw preview information on frame."""
        height, width = frame.shape[:2]
        
        # Status text
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        control_status = "ON" if self.gaze_control_enabled else "OFF"
        
        cv2.putText(frame, f"Face: {'Detected' if face_detected else 'Not Found'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Gaze Control: {control_status}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}°  Yaw: {yaw:.1f}°", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show calculated screen coordinates with screen index
        if face_detected:
            screen_x, screen_y, screen_idx = self.gaze_mapper.gaze_to_screen(pitch, yaw)
            if screen_x is not None and screen_y is not None:
                cv2.putText(frame, f"Screen: ({screen_x}, {screen_y}) [Screen {screen_idx}]", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "SPACE: Toggle Control  P: Toggle Preview  Q: Quit", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def cleanup(self):
        """Clean up resources."""
        if self.stop_event:
            self.stop_event.set()
        
        # Wait for threads to finish
        if self.cam_thread and self.cam_thread.is_alive():
            logging.info("Waiting for camera thread to stop...")
            self.cam_thread.join(timeout=2)
        if self.inf_thread and self.inf_thread.is_alive():
            logging.info("Waiting for inference thread to stop...")
            self.inf_thread.join(timeout=2)
        
        self.uart_reader.stop()
        self.camera_manager.release()
        cv2.destroyAllWindows()
        
        logging.info("\n=== Pipeline Statistics ===")
        if self.camera_fps:
            logging.info(f"Camera FPS: {self.camera_fps.get_fps():.2f}")
        if self.inference_fps:
            logging.info(f"Inference FPS: {self.inference_fps.get_fps():.2f}")
        if self.display_fps:
            logging.info(f"Display FPS: {self.display_fps.get_fps():.2f}")
        logging.info("Application cleaned up")


def main():
    """Main entry point."""
    import os
    
    parser = argparse.ArgumentParser(description="Simple gaze-controlled mouse pointer")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--calibration", type=str, default=None,
                       help="Calibration file path (default: auto-detect most recent)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (logs movements without moving mouse)")
    parser.add_argument("--timeout", type=float, default=3.0,
                       help="Run duration in minutes (default: 3.0)")
    parser.add_argument("--window-size", type=int, default=3,
                       help="Mouse smoothing window size (SMA, default: 3)")
    parser.add_argument("--min-movement", type=int, default=30,
                       help="Dead zone radius in pixels (default: 30)")
    args = parser.parse_args()
    
    # Auto-detect calibration file if not specified
    calibration_file = args.calibration
    if calibration_file is None:
        # Look for most recent cocoa calibration file
        results_dir = "calibration_results"
        if os.path.exists(results_dir):
            files = [f for f in os.listdir(results_dir) 
                    if f.startswith("cocoa_calibration_") and f.endswith(".csv")]
            if files:
                files.sort(reverse=True)  # Most recent first
                calibration_file = os.path.join(results_dir, files[0])
                print(f"Auto-detected calibration: {files[0]}")
            else:
                calibration_file = "cocoa_calibration.csv"
        else:
            calibration_file = "cocoa_calibration.csv"
    
    debug_mode = args.debug
    timeout_minutes = args.timeout
    window_size = max(1, args.window_size)  # Ensure at least 1
    min_movement = max(0, args.min_movement)  # Ensure non-negative
    
    print("=" * 60)
    print("Simple Gaze Control (Calibration-Based)")
    print("=" * 60)
    print("Using calibration-based linear regression models")
    print(f"Calibration file: {calibration_file}")
    print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    print(f"Run duration: {timeout_minutes} minutes")
    print(f"Mouse smoothing: SMA window_size={window_size}, min movement={min_movement}px")
    print()
    
    try:
        app = SimpleGazeApp(args.camera, debug_mode, timeout_minutes, 
                          window_size, min_movement, 
                          calibration_file=calibration_file)
        app.run()
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
