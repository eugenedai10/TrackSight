#!/usr/bin/env python3
"""
Native Cocoa Gaze Calibration Application

Uses native macOS Cocoa/AppKit for multi-screen calibration.
Creates separate windows for each screen for reliable multi-screen support.
All coordinates use Cocoa's native bottom-left origin system.
"""

import logging
import time
import numpy as np
from datetime import datetime
import threading
import queue
from collections import deque
from AppKit import (
    NSApplication, NSWindow, NSView, NSScreen, NSColor, NSFont, NSMakeRect,
    NSBorderlessWindowMask, NSBackingStoreBuffered, NSTimer,
    NSBezierPath, NSAttributedString,
    NSFontAttributeName, NSForegroundColorAttributeName,
    NSMakePoint, NSEvent, NSKeyDownMask
)
from Foundation import NSObject
import objc

# Import existing components
from gaze_processor import GazeProcessor
from calibration.camera_utils import CameraManager

# Setup logging
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


def gaze_estimation_thread(gaze_processor, frame_queue, result_queue, stop_event, inference_fps):
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
        
        inference_fps.update()
        
        try:
            # Send results to main thread
            result_queue.put((timestamp, pitch, yaw, face_detected), timeout=0.01)
        except queue.Full:
            pass  # Drop if main thread is slow
    
    logging.info("Gaze estimation thread stopped")


class CalibrationView(NSView):
    """Custom NSView for rendering calibration interface on one screen."""
    
    def initWithFrame_(self, frame):
        """Initialize the view."""
        self = objc.super(CalibrationView, self).initWithFrame_(frame)
        if self is None:
            return None
        
        # Parent app reference and screen assignment
        self.parent_app = None
        self.screen_index = 0
        
        # Visual settings
        self.dot_radius = 15.0
        self.bg_color = NSColor.blackColor()
        self.dot_color = NSColor.whiteColor()
        self.text_color = NSColor.whiteColor()
        self.green_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 1.0, 0.0, 1.0)
        self.gray_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.5, 0.5, 0.5, 1.0)
        
        # Fonts
        self.title_font = NSFont.systemFontOfSize_(48.0)
        self.text_font = NSFont.systemFontOfSize_(32.0)
        self.small_font = NSFont.systemFontOfSize_(24.0)
        self.tiny_font = NSFont.systemFontOfSize_(16.0)
        
        return self
    
    def acceptsFirstResponder(self):
        """Accept keyboard events."""
        return True
    
    def keyDown_(self, event):
        """Handle keyboard events."""
        if not self.parent_app:
            return
            
        key = event.charactersIgnoringModifiers()
        modifiers = event.modifierFlags()
        logging.info(f"Key pressed: {repr(key)}, modifiers: {modifiers}")
        
        # Check for Cmd+Q (Command modifier = 1048576)
        if key.lower() == 'q' and (modifiers & 1048576):
            logging.info("Cmd+Q pressed - exiting")
            self.parent_app.running = False
            NSApplication.sharedApplication().terminate_(None)
        
        elif key == '\x1b':  # ESC
            logging.info("ESC pressed - exiting")
            self.parent_app.running = False
            NSApplication.sharedApplication().terminate_(None)
        
        elif self.parent_app.state == 'welcome' and key == ' ':
            logging.info("SPACE pressed - starting calibration")
            self.parent_app._start_countdown()
        
        elif self.parent_app.state == 'completion':
            logging.info("Key pressed on completion - exiting")
            self.parent_app.running = False
            NSApplication.sharedApplication().terminate_(None)
    
    def mouseDown_(self, event):
        """Handle mouse clicks."""
        if not self.parent_app:
            return
            
        logging.info("Mouse clicked")
        
        # Ensure window has focus
        window = self.window()
        if window:
            window.makeKeyWindow()
            window.makeFirstResponder_(self)
        
        # In welcome state, start calibration
        if self.parent_app.state == 'welcome':
            logging.info("Click in welcome state - starting calibration")
            self.parent_app._start_countdown()
        
        # In completion state, exit
        elif self.parent_app.state == 'completion':
            logging.info("Click in completion state - exiting")
            self.parent_app.running = False
            NSApplication.sharedApplication().terminate_(None)
    
    def drawRect_(self, rect):
        """Draw the current state."""
        if not self.parent_app:
            return
            
        # Fill background
        self.bg_color.set()
        NSBezierPath.fillRect_(rect)
        
        state = self.parent_app.state
        if state == 'welcome':
            self._draw_welcome_screen()
        elif state in ('countdown', 'dot'):
            self._draw_dot_state()
        elif state == 'completion':
            self._draw_completion_screen()
    
    def _draw_welcome_screen(self):
        """Draw welcome screen."""
        bounds = self.bounds()
        center_x = bounds.size.width / 2
        center_y = bounds.size.height / 2
        
        # Title
        self._draw_centered_text("Gaze Calibration", self.title_font, self.text_color,
                                center_x, center_y + 150)
        
        # Instructions
        num_dots = len(self.parent_app.dot_positions)
        num_screens = len(self.parent_app.screen_info)
        
        self._draw_centered_text(f"{num_dots} dots across {num_screens} screen(s)",
                                self.text_font, self.text_color, center_x, center_y + 80)
        self._draw_centered_text("Look at each dot when it appears",
                                self.text_font, self.text_color, center_x, center_y + 50)
        self._draw_centered_text("Keep your head still",
                                self.text_font, self.text_color, center_x, center_y + 20)
        
        # Start instruction
        self._draw_centered_text("Click or press SPACE to start",
                                self.text_font, self.green_color, center_x, center_y - 60)
        self._draw_centered_text("Press ESC to exit",
                                self.small_font, self.gray_color, center_x, center_y - 100)
    
    def _draw_dot_state(self):
        """Draw countdown or dot display."""
        # Draw progress indicator at top center (on all screens)
        bounds = self.bounds()
        center_x = bounds.size.width / 2
        total_dots = len(self.parent_app.dot_positions)
        current = self.parent_app.current_dot_index + 1
        progress_text = f"Point {current} of {total_dots}"
        self._draw_centered_text(progress_text, self.text_font, self.text_color,
                                center_x, bounds.size.height - 80)
        
        # Draw exit reminder at bottom center (on all screens)
        self._draw_centered_text("Press ESC or Cmd+Q to cancel",
                                self.small_font, self.gray_color,
                                center_x, 40)
        
        # Check if current dot is on this screen
        current_idx = self.parent_app.current_dot_index
        if current_idx >= len(self.parent_app.dot_positions):
            return
            
        pos_x, pos_y, screen_idx = self.parent_app.dot_positions[current_idx]
        
        # Only draw on the screen that has the current dot
        if screen_idx != self.screen_index:
            return
        
        # Get window origin to convert to view coordinates
        window = self.window()
        if not window:
            return
            
        window_origin_x = window.frame().origin.x
        window_origin_y = window.frame().origin.y
        window_width = window.frame().size.width
        window_height = window.frame().size.height
        
        # Convert from Cocoa screen coords to window-relative coords
        view_x = pos_x - window_origin_x
        view_y = pos_y - window_origin_y
        
        
        # Draw the dot
        self.dot_color.set()
        path = NSBezierPath.bezierPath()
        path.appendBezierPathWithOvalInRect_(
            NSMakeRect(view_x - self.dot_radius, view_y - self.dot_radius,
                      self.dot_radius * 2, self.dot_radius * 2)
        )
        path.fill()
        
        # Draw timer on dot if in dot state
        if self.parent_app.state == 'dot':
            timer_text = f"{self.parent_app.dot_timer:.1f}s"
            self._draw_centered_text(timer_text, self.tiny_font, self.gray_color,
                                    view_x, view_y)
    
    def _draw_completion_screen(self):
        """Draw completion screen."""
        bounds = self.bounds()
        center_x = bounds.size.width / 2
        center_y = bounds.size.height / 2
        
        # Title
        self._draw_centered_text("Calibration Complete!", self.title_font, self.green_color,
                                center_x, center_y + 100)
        
        # Stats
        num_dots = len(self.parent_app.calibration_results)
        self._draw_centered_text(f"Collected {num_dots} points",
                                self.text_font, self.text_color, center_x, center_y + 40)
        
        # Visualization message
        self._draw_centered_text("Launching visualization...",
                                self.text_font, self.text_color, center_x, center_y - 20)
    
    def _draw_centered_text(self, text, font, color, center_x, center_y):
        """Draw centered text at position."""
        attrs = {
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: color
        }
        
        attr_string = NSAttributedString.alloc().initWithString_attributes_(text, attrs)
        size = attr_string.size()
        
        point = NSMakePoint(center_x - size.width / 2, center_y - size.height / 2)
        attr_string.drawAtPoint_(point)


class CocoaCalibrationApp(NSObject):
    """Main calibration application using native Cocoa with multiple windows."""
    
    def init(self):
        """Initialize the application."""
        self = objc.super(CocoaCalibrationApp, self).init()
        if self is None:
            return None
        
        # Components
        self.windows = []  # Multiple windows, one per screen
        self.views = []    # Multiple views, one per screen
        self.camera_manager = None
        self.gaze_processor = None
        self.event_monitor = None  # Global event monitor
        
        # Calibration state
        self.screen_info = []
        self.dot_positions = []
        self.calibration_results = []
        self.running = True
        
        # State machine
        self.state = 'welcome'  # welcome, countdown, dot, completion
        self.current_dot_index = 0
        self.countdown_value = 3
        self.dot_timer = 0.0
        
        # Timing
        self.dot_duration = 5.0  # seconds
        self.countdown_duration = 3.0  # seconds
        self.timer = None
        self.state_start_time = 0
        
        # Current gaze data
        self.current_gaze_data = []
        
        # Threading components
        self.frame_queue = None
        self.result_queue = None
        self.stop_event = None
        self.camera_fps = None
        self.inference_fps = None
        self.cam_thread = None
        self.inf_thread = None
        
        return self
    
    def setup(self):
        """Set up the calibration system."""
        logging.info("=== Setting up Multi-Window Cocoa Calibration ===")
        
        # Detect screens
        self._detect_screens()
        
        # Calculate dot positions
        self._calculate_dot_positions()
        
        # Create windows for each screen
        self._create_windows()
        
        # Initialize camera and gaze processor
        self._initialize_camera_and_gaze()
        
        logging.info("Setup complete")
        return True
    
    def _detect_screens(self):
        """Detect all screens and their configuration."""
        screens = NSScreen.screens()
        self.screen_info = []
        
        logging.info(f"Detected {len(screens)} screen(s)")
        
        # Build screen data first
        screen_list = []
        for screen in screens:
            frame = screen.frame()
            
            screen_data = {
                'origin_x': frame.origin.x,
                'origin_y': frame.origin.y,
                'width': frame.size.width,
                'height': frame.size.height,
                'is_main': screen == NSScreen.mainScreen()
            }
            screen_list.append(screen_data)
        
        # Sort screens: main screen first, then by x position
        screen_list.sort(key=lambda s: (not s['is_main'], s['origin_x']))
        
        # Assign indices after sorting
        for idx, screen_data in enumerate(screen_list):
            screen_data['index'] = idx
            self.screen_info.append(screen_data)
            
            logging.info(f"Screen {idx}: origin=({screen_data['origin_x']:.0f}, {screen_data['origin_y']:.0f}), "
                        f"size={screen_data['width']:.0f}x{screen_data['height']:.0f}, "
                        f"is_main={screen_data['is_main']}")
    
    def _calculate_dot_positions(self):
        """Calculate calibration dot positions for all screens."""
        self.dot_positions = []
        inset = 15  # pixels from edge
        
        for screen in self.screen_info:
            screen_idx = screen['index']
            x0 = screen['origin_x']
            y0 = screen['origin_y']
            w = screen['width']
            h = screen['height']
            
            # 4 corners per screen (in Cocoa coordinates)
            # Note: For top corners, need to account for actual window bounds
            corners = [
                (x0 + inset, y0 + inset, screen_idx),                    # Bottom-left
                (x0 + w - inset, y0 + inset, screen_idx),                # Bottom-right  
                (x0 + inset, y0 + (h - 1) - inset, screen_idx),          # Top-left
                (x0 + w - inset, y0 + (h - 1) - inset, screen_idx),      # Top-right
            ]
            
            self.dot_positions.extend(corners)
            
            logging.info(f"Screen {screen_idx} calibration points:")
            for i, (x, y, _) in enumerate(corners):
                logging.info(f"  Corner {i+1}: ({x:.0f}, {y:.0f})")
        
        logging.info(f"Total calibration points: {len(self.dot_positions)}")
    
    def _create_windows(self):
        """Create a borderless window for each screen."""
        for screen in self.screen_info:
            screen_idx = screen['index']
            x = screen['origin_x']
            y = screen['origin_y']
            w = screen['width']
            h = screen['height']
            
            # Create window for this screen
            frame = NSMakeRect(x, y, w, h)
            
            window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                NSBorderlessWindowMask,
                NSBackingStoreBuffered,
                False
            )
            
            window.setBackgroundColor_(NSColor.blackColor())
            window.setLevel_(25)  # NSStatusWindowLevel - above menu bar
            window.setOpaque_(True)
            
            # Create view for this window
            view = CalibrationView.alloc().initWithFrame_(frame)
            view.parent_app = self
            view.screen_index = screen_idx
            
            window.setContentView_(view)
            window.makeKeyAndOrderFront_(None)
            
            self.windows.append(window)
            self.views.append(view)
            
            logging.info(f"Created window for screen {screen_idx}")
        
        # Make first window key
        if self.windows:
            self.windows[0].makeKeyWindow()
            self.windows[0].makeFirstResponder_(self.views[0])
    
    def _initialize_camera_and_gaze(self):
        """Initialize camera and gaze processing with parallel threads."""
        logging.info("Initializing camera and gaze estimation (Parallel Mode)...")
        
        self.camera_manager = CameraManager(camera_index=0)
        self.camera_manager.initialize()
        
        self.gaze_processor = GazeProcessor()
        
        # Create queues and FPS counters for parallel processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)  # Larger queue for calibration data
        self.stop_event = threading.Event()
        self.camera_fps = FPSCounter()
        self.inference_fps = FPSCounter()
        
        # Start background threads
        self.cam_thread = threading.Thread(
            target=camera_thread,
            args=(self.camera_manager, self.frame_queue, self.stop_event, self.camera_fps),
            daemon=True
        )
        
        self.inf_thread = threading.Thread(
            target=gaze_estimation_thread,
            args=(self.gaze_processor, self.frame_queue, self.result_queue, self.stop_event, self.inference_fps),
            daemon=True
        )
        
        self.cam_thread.start()
        self.inf_thread.start()
        
        logging.info("Camera and gaze estimation initialized with parallel threads")
        logging.info("Pipeline: Camera -> Inference -> Main Thread (Calibration)")
    
    def _handle_global_key(self, event):
        """Handle global keyboard events."""
        key = event.charactersIgnoringModifiers()
        modifiers = event.modifierFlags()
        
        # Handle ESC - always exit
        if key == '\x1b':
            logging.info("Global: ESC pressed - exiting")
            self.running = False
            NSApplication.sharedApplication().terminate_(None)
        
        # Handle Cmd+Q - always exit
        elif key.lower() == 'q' and (modifiers & 1048576):
            logging.info("Global: Cmd+Q pressed - exiting")
            self.running = False
            NSApplication.sharedApplication().terminate_(None)
        
        # Handle SPACE - start calibration if in welcome state
        elif key == ' ' and self.state == 'welcome':
            logging.info("Global: SPACE pressed - starting calibration")
            self._start_countdown()
    
    def start(self):
        """Start the calibration process."""
        # Set initial state
        self.state = 'welcome'
        self._refresh_all_views()
        
        # Set up global event monitor for ESC and Cmd+Q
        self.event_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSKeyDownMask,
            self._handle_global_key
        )
        logging.info("Global keyboard monitor installed")
        
        # Set up timer for updates
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.033,  # ~30 FPS
            self,
            'timerFired:',
            None,
            True
        )
        
        # Activate application
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        
        logging.info("Calibration started - showing welcome screen")
        logging.info("Click or press SPACE to begin, ESC or Cmd+Q to exit")
    
    def _refresh_all_views(self):
        """Refresh all views."""
        for view in self.views:
            view.setNeedsDisplay_(True)
    
    def timerFired_(self, timer):
        """Timer callback for updates."""
        if not self.running:
            if self.timer:
                self.timer.invalidate()
                self.timer = None
            return
        
        # Update based on current state
        if self.state == 'countdown':
            elapsed = time.time() - self.state_start_time
            remaining = self.countdown_duration - elapsed
            
            if remaining <= 0:
                self._start_dot_display()
            else:
                self.countdown_value = int(remaining) + 1
                self._refresh_all_views()
        
        elif self.state == 'dot':
            elapsed = time.time() - self.state_start_time
            remaining = self.dot_duration - elapsed
            
            # Collect gaze data
            self._collect_gaze_data()
            
            if remaining <= 0:
                self._finish_dot_display()
            else:
                self.dot_timer = remaining
                self._refresh_all_views()
    
    def _start_countdown(self):
        """Start countdown for next dot."""
        self.state = 'countdown'
        self.countdown_value = 3
        self.state_start_time = time.time()
        self._refresh_all_views()
        
        # Re-establish keyboard focus
        if self.windows:
            self.windows[0].makeKeyWindow()
            self.windows[0].makeFirstResponder_(self.views[0])
        
        logging.info(f"Starting countdown for dot {self.current_dot_index + 1}")
    
    def _start_dot_display(self):
        """Start displaying current dot."""
        self.state = 'dot'
        self.dot_timer = self.dot_duration
        self.state_start_time = time.time()
        self.current_gaze_data = []
        self._refresh_all_views()
        
        # Re-establish keyboard focus
        if self.windows:
            self.windows[0].makeKeyWindow()
            self.windows[0].makeFirstResponder_(self.views[0])
        
        logging.info(f"Showing dot {self.current_dot_index + 1}")
    
    def _collect_gaze_data(self):
        """Collect gaze data from parallel processing queue."""
        # Get all available gaze results from queue (non-blocking)
        while True:
            try:
                timestamp, pitch, yaw, face_detected = self.result_queue.get_nowait()
                
                if face_detected:
                    self.current_gaze_data.append((pitch, yaw))
            except queue.Empty:
                break
    
    def _finish_dot_display(self):
        """Finish current dot and move to next."""
        # Save calibration data
        if self.current_dot_index < len(self.dot_positions):
            pos_x, pos_y, screen_idx = self.dot_positions[self.current_dot_index]
            
            # Calculate statistics
            if self.current_gaze_data:
                pitches = [p for p, y in self.current_gaze_data]
                yaws = [y for p, y in self.current_gaze_data]
                
                result = {
                    'screen_index': screen_idx,
                    'position_index': self.current_dot_index,
                    'cocoa_x': pos_x,
                    'cocoa_y': pos_y,
                    'timestamp': datetime.now().isoformat(),
                    'pitch_avg': np.mean(pitches),
                    'yaw_avg': np.mean(yaws),
                    'sample_count': len(self.current_gaze_data)
                }
                
                self.calibration_results.append(result)
                
                logging.info(f"Dot {self.current_dot_index + 1}: "
                           f"Screen={screen_idx}, Pitch={result['pitch_avg']:.1f}°, "
                           f"Yaw={result['yaw_avg']:.1f}°, Samples={result['sample_count']}")
        
        # Move to next dot or finish
        self.current_dot_index += 1
        
        if self.current_dot_index < len(self.dot_positions):
            self._start_countdown()
        else:
            self._finish_calibration()
    
    def _finish_calibration(self):
        """Finish calibration and show results."""
        self._save_results()
        
        self.state = 'completion'
        self._refresh_all_views()
        
        logging.info("Calibration complete!")
        logging.info("Launching visualization...")
        
        # Launch visualization in separate process
        import subprocess
        import sys
        subprocess.Popen([sys.executable, 'visualize_cocoa_calibration.py', self.saved_filename])
        
        # Exit calibration immediately
        self.running = False
        NSApplication.sharedApplication().terminate_(None)
    
    def _save_results(self):
        """Save calibration results to CSV."""
        import os
        os.makedirs("calibration_results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_results/cocoa_calibration_{timestamp}.csv"
        
        with open(filename, 'w') as f:
            f.write("screen_index,position_index,cocoa_x,cocoa_y,timestamp,pitch_avg,yaw_avg,sample_count\n")
            
            for result in self.calibration_results:
                f.write(f"{result['screen_index']},")
                f.write(f"{result['position_index']},")
                f.write(f"{result['cocoa_x']:.1f},{result['cocoa_y']:.1f},")
                f.write(f"{result['timestamp']},")
                f.write(f"{result['pitch_avg']:.2f},{result['yaw_avg']:.2f},")
                f.write(f"{result['sample_count']}\n")
        
        # Store filename for visualization
        self.saved_filename = filename
        
        logging.info(f"Results saved to {filename}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.timer:
            self.timer.invalidate()
            self.timer = None
        
        if self.event_monitor:
            NSEvent.removeMonitor_(self.event_monitor)
            self.event_monitor = None
            logging.info("Global keyboard monitor removed")
        
        # Stop background threads
        if self.stop_event:
            self.stop_event.set()
            logging.info("Stopping background threads...")
        
        if self.cam_thread and self.cam_thread.is_alive():
            self.cam_thread.join(timeout=2)
        if self.inf_thread and self.inf_thread.is_alive():
            self.inf_thread.join(timeout=2)
        
        if self.camera_manager:
            self.camera_manager.release()
        
        # Log final statistics
        if self.camera_fps and self.inference_fps:
            logging.info("\n=== Parallel Pipeline Statistics ===")
            logging.info(f"Camera FPS: {self.camera_fps.get_fps():.2f}")
            logging.info(f"Inference FPS: {self.inference_fps.get_fps():.2f}")
        
        logging.info("Cleanup complete")


def main():
    """Main entry point."""
    print("=" * 70)
    print("Multi-Window Cocoa Gaze Calibration")
    print("=" * 70)
    print()
    print("Features:")
    print("  - Native Cocoa coordinates (bottom-left origin)")
    print("  - Separate window per screen (reliable multi-screen)")
    print("  - 4 corner points per screen")
    print("  - Clean, minimal UI")
    print()
    
    # Create application
    app = NSApplication.sharedApplication()
    
    # Create calibration app
    calibration_app = CocoaCalibrationApp.alloc().init()
    app.setDelegate_(calibration_app)
    
    # Setup
    if not calibration_app.setup():
        logging.error("Failed to set up calibration")
        return
    
    # Start calibration
    calibration_app.start()
    
    # Run application
    app.run()
    
    # Cleanup
    calibration_app.cleanup()
    
    print()
    print("Calibration complete!")


if __name__ == "__main__":
    main()
