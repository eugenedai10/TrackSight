#!/usr/bin/env python3
"""
Native Cocoa Calibration Visualization

Displays calibration data from cocoa_calibration.csv using native macOS AppKit.
Shows screens, calibration points, and pitch/yaw values.
"""

import logging
import csv
import pandas as pd
import numpy as np
from AppKit import (
    NSApplication, NSWindow, NSView, NSScreen, NSColor, NSFont, NSMakeRect,
    NSTitledWindowMask, NSClosableWindowMask, NSMiniaturizableWindowMask,
    NSBackingStoreBuffered, NSApplicationActivationPolicyRegular,
    NSBezierPath, NSAttributedString,
    NSFontAttributeName, NSForegroundColorAttributeName,
    NSMakePoint
)
from Foundation import NSObject
import objc

# Import calibration utilities
from calibration.data_utils import (
    check_data_consistency,
    get_screen_bounds,
    generate_test_data
)
from calibration.model_utils import (
    fit_linear_models,
    predict_coordinates
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class VisualizationView(NSView):
    """Custom NSView for rendering calibration visualization."""
    
    def initWithFrame_(self, frame):
        """Initialize the view."""
        self = objc.super(VisualizationView, self).initWithFrame_(frame)
        if self is None:
            return None
        
        # Data
        self.calibration_data = []
        self.screen_bounds = {}
        self.world_bounds = None  # Will be set by parent app
        self.error_details = []  # Error information
        self.model_params = {}  # Regression models
        self.test_data = []  # Test predictions
        self.analysis_screen_bounds = []  # Screen bounds for predictions
        
        # Colors
        self.bg_color = NSColor.blackColor()
        self.valid_dot_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 1.0, 0.4, 1.0)  # Green
        self.error_dot_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.2, 0.2, 1.0)  # Red
        self.error_line_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.2, 0.2, 1.0)  # Red
        self.screen_colors = [
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.3, 0.5, 1.0),  # Blue
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.3, 0.5, 0.3, 1.0),  # Green
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.5, 0.3, 0.3, 1.0),  # Red
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.5, 0.4, 0.2, 1.0),  # Orange
        ]
        self.dot_color = NSColor.whiteColor()
        self.text_color = NSColor.whiteColor()
        
        # Fonts
        self.font_small = NSFont.systemFontOfSize_(12.0)
        
        # Display toggles
        self.show_test_points = True  # Toggle for test predictions
        
        return self
    
    def acceptsFirstResponder(self):
        """Accept keyboard events."""
        return True
    
    def keyDown_(self, event):
        """Handle keyboard events."""
        key = event.charactersIgnoringModifiers()
        
        if key == '\x1b':  # ESC
            logging.info("ESC pressed - exiting")
            NSApplication.sharedApplication().terminate_(None)
        elif key.lower() == 't':  # Toggle test points
            self.show_test_points = not self.show_test_points
            logging.info(f"Test points: {'ON' if self.show_test_points else 'OFF'}")
            self.setNeedsDisplay_(True)  # Trigger redraw
    
    def drawRect_(self, rect):
        """Draw the visualization."""
        # Fill background
        self.bg_color.set()
        NSBezierPath.fillRect_(rect)
        
        # Draw screens
        self._draw_screens()
        
        # Draw error lines (behind points)
        self._draw_error_lines()
        
        # Draw calibration points
        self._draw_calibration_points()
        
        # Draw test predictions (on top)
        self._draw_test_predictions()
        
        # Draw legend (overlay on top)
        self._draw_legend()
    
    def _world_to_view(self, world_x, world_y):
        """Convert world coordinates to view coordinates with scaling."""
        if not self.world_bounds:
            return (world_x, world_y)
        
        bounds = self.bounds()
        
        # Add padding
        padding = 40
        view_width = bounds.size.width - 2 * padding
        view_height = bounds.size.height - 2 * padding
        
        # Calculate scale to fit (maintain aspect ratio)
        scale_x = view_width / self.world_bounds['width']
        scale_y = view_height / self.world_bounds['height']
        scale = min(scale_x, scale_y)
        
        # Calculate offset to center
        scaled_width = self.world_bounds['width'] * scale
        scaled_height = self.world_bounds['height'] * scale
        offset_x = padding + (view_width - scaled_width) / 2
        offset_y = padding + (view_height - scaled_height) / 2
        
        # Convert
        view_x = offset_x + (world_x - self.world_bounds['min_x']) * scale
        view_y = offset_y + (world_y - self.world_bounds['min_y']) * scale
        
        return (view_x, view_y)
    
    def _draw_screens(self):
        """Draw screen rectangles."""
        for screen_idx, bounds in self.screen_bounds.items():
            # Convert corners to view coordinates
            bottom_left = self._world_to_view(bounds['min_x'], bounds['min_y'])
            top_right = self._world_to_view(bounds['max_x'], bounds['max_y'])
            
            view_x = bottom_left[0]
            view_y = bottom_left[1]
            view_w = top_right[0] - bottom_left[0]
            view_h = top_right[1] - bottom_left[1]
            
            # Draw rectangle
            color = self.screen_colors[screen_idx % len(self.screen_colors)]
            color.set()
            
            rect = NSMakeRect(view_x, view_y, view_w, view_h)
            path = NSBezierPath.bezierPathWithRect_(rect)
            path.setLineWidth_(3.0)
            path.stroke()
            
            # Draw screen label
            label = f"Screen {screen_idx}"
            self._draw_text(label, view_x + view_w / 2, view_y + view_h / 2,
                          self.text_color, centered=True)
    
    def _draw_error_lines(self):
        """Draw red lines between error points."""
        if not self.error_details:
            return
        
        self.error_line_color.set()
        
        for error in self.error_details:
            # Get the two points involved in this error
            point_indices = error['points']
            
            if len(point_indices) >= 2:
                point1 = self.calibration_data[point_indices[0]]
                point2 = self.calibration_data[point_indices[1]]
                
                # Convert to view coordinates
                pos1 = self._world_to_view(point1['cocoa_x'], point1['cocoa_y'])
                pos2 = self._world_to_view(point2['cocoa_x'], point2['cocoa_y'])
                
                # Draw line
                path = NSBezierPath.bezierPath()
                path.moveToPoint_(NSMakePoint(pos1[0], pos1[1]))
                path.lineToPoint_(NSMakePoint(pos2[0], pos2[1]))
                path.setLineWidth_(2.0)
                path.stroke()
                
                # Draw error value at midpoint
                mid_x = (pos1[0] + pos2[0]) / 2
                mid_y = (pos1[1] + pos2[1]) / 2
                error_text = f"{error['diff_abs']:.1f}"
                self._draw_text(error_text, mid_x, mid_y, self.error_line_color, centered=True)
    
    def _draw_calibration_points(self):
        """Draw calibration points with labels."""
        for point in self.calibration_data:
            cocoa_x = point['cocoa_x']
            cocoa_y = point['cocoa_y']
            pitch = point['pitch_avg']
            yaw = point['yaw_avg']
            has_error = point.get('has_error', False)
            
            # Convert to view coordinates
            view_x, view_y = self._world_to_view(cocoa_x, cocoa_y)
            
            # Choose color based on error status
            if has_error:
                dot_color = self.error_dot_color
                outline_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.4, 0.4, 1.0)
            else:
                dot_color = self.valid_dot_color
                outline_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.4, 1.0, 0.6, 1.0)
            
            # Draw dot
            dot_color.set()
            radius = 8.0
            oval_rect = NSMakeRect(view_x - radius, view_y - radius, 
                                  radius * 2, radius * 2)
            path = NSBezierPath.bezierPathWithOvalInRect_(oval_rect)
            path.fill()
            
            # Draw outline
            outline_color.set()
            path.setLineWidth_(2.0)
            path.stroke()
            
            # Draw pitch/yaw and coordinate labels
            pitch_label = f"P:{pitch:.1f}°"
            yaw_label = f"Y:{yaw:.1f}°"
            coord_label = f"({cocoa_x:.0f}, {cocoa_y:.0f})"
            
            # Position labels to the right of the dot
            label_x = view_x + 12
            label_y = view_y + 6
            
            self._draw_text(pitch_label, label_x, label_y, self.text_color)
            self._draw_text(yaw_label, label_x, label_y - 14, self.text_color)
            self._draw_text(coord_label, label_x, label_y - 28, self.text_color)
    
    def _draw_test_predictions(self):
        """Draw test prediction points with raw predictions from each screen model."""
        # Check if test points should be shown
        if not self.show_test_points:
            return
        
        if not self.test_data or not self.model_params or not self.analysis_screen_bounds:
            return
        
        # Colors for raw predictions per screen
        raw_colors = [
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.7, 0.4, 1.0),  # Orange - Screen 0
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.4, 0.9, 0.9, 1.0),  # Cyan - Screen 1
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 0.4, 1.0),  # Yellow - Screen 2
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.4, 0.8, 1.0),  # Magenta - Screen 3
        ]
        
        capped_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.8, 0.4, 1.0, 1.0)  # Purple
        line_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.6, 0.6, 0.6, 1.0)  # Gray
        font_tiny = NSFont.systemFontOfSize_(10.0)
        
        for i, (pitch, yaw) in enumerate(self.test_data):
            try:
                # Calculate CAPPED prediction (final selected position)
                capped_x, capped_y, selected_screen = predict_coordinates(
                    pitch, yaw, self.model_params, self.analysis_screen_bounds
                )
                capped_view_x, capped_view_y = self._world_to_view(capped_x, capped_y)
                
                # Draw raw predictions from each screen's model
                for screen_idx in sorted(self.model_params.keys()):
                    params = self.model_params[screen_idx]
                    
                    # Calculate raw position using this screen's model
                    raw_x = (pitch - params['pitch']['intercept']) / params['pitch']['slope']
                    raw_y = (yaw - params['yaw']['intercept']) / params['yaw']['slope']
                    raw_view_x, raw_view_y = self._world_to_view(raw_x, raw_y)
                    
                    # Check if raw differs from capped (different model selected)
                    differs = (abs(raw_x - capped_x) > 0.1 or abs(raw_y - capped_y) > 0.1)
                    
                    # Draw connecting line if different
                    if differs:
                        line_color.set()
                        path = NSBezierPath.bezierPath()
                        path.moveToPoint_(NSMakePoint(raw_view_x, raw_view_y))
                        path.lineToPoint_(NSMakePoint(capped_view_x, capped_view_y))
                        path.setLineWidth_(1.0)
                        path.stroke()
                    
                    # Draw raw prediction dot
                    color = raw_colors[screen_idx % len(raw_colors)]
                    color.set()
                    radius = 5.0
                    oval_rect = NSMakeRect(raw_view_x - radius, raw_view_y - radius, 
                                          radius * 2, radius * 2)
                    path = NSBezierPath.bezierPathWithOvalInRect_(oval_rect)
                    path.fill()
                    
                    # White outline
                    NSColor.whiteColor().set()
                    path.setLineWidth_(1.0)
                    path.stroke()
                
                # Draw capped prediction diamond (on top)
                size = 8
                points = [
                    NSMakePoint(capped_view_x, capped_view_y - size),      # Top
                    NSMakePoint(capped_view_x + size, capped_view_y),      # Right
                    NSMakePoint(capped_view_x, capped_view_y + size),      # Bottom
                    NSMakePoint(capped_view_x - size, capped_view_y),      # Left
                ]
                
                capped_color.set()
                path = NSBezierPath.bezierPath()
                path.moveToPoint_(points[0])
                for point in points[1:]:
                    path.lineToPoint_(point)
                path.closePath()
                path.fill()
                
                # Draw white outline
                NSColor.whiteColor().set()
                path.setLineWidth_(1.5)
                path.stroke()
                
                # Draw detailed labels - only for first 10 to avoid clutter
                if i < 10:
                    label_text = f"T{i+1}"
                    pitch_text = f"P:{pitch:.1f}°"
                    yaw_text = f"Y:{yaw:.1f}°"
                    coord_text = f"({capped_x:.0f}, {capped_y:.0f})"
                    
                    # Position labels to the right
                    label_x = capped_view_x + 12
                    label_y = capped_view_y + 8
                    
                    # Draw label
                    attrs = {
                        NSFontAttributeName: font_tiny,
                        NSForegroundColorAttributeName: NSColor.colorWithCalibratedRed_green_blue_alpha_(0.8, 0.8, 1.0, 1.0)
                    }
                    attr_string = NSAttributedString.alloc().initWithString_attributes_(label_text, attrs)
                    attr_string.drawAtPoint_(NSMakePoint(label_x, label_y))
                    
                    # Draw pitch
                    attrs[NSForegroundColorAttributeName] = NSColor.whiteColor()
                    attr_string = NSAttributedString.alloc().initWithString_attributes_(pitch_text, attrs)
                    attr_string.drawAtPoint_(NSMakePoint(label_x, label_y - 11))
                    
                    # Draw yaw
                    attr_string = NSAttributedString.alloc().initWithString_attributes_(yaw_text, attrs)
                    attr_string.drawAtPoint_(NSMakePoint(label_x, label_y - 22))
                    
                    # Draw coordinates
                    attr_string = NSAttributedString.alloc().initWithString_attributes_(coord_text, attrs)
                    attr_string.drawAtPoint_(NSMakePoint(label_x, label_y - 33))
                
            except (ValueError, OverflowError, KeyError, ZeroDivisionError):
                # Skip if prediction fails
                pass
    
    def _draw_legend(self):
        """Draw legend explaining visual elements."""
        bounds = self.bounds()
        
        # Position at bottom-left
        legend_x = 20
        legend_y = 20
        font_medium = NSFont.boldSystemFontOfSize_(14.0)
        font_small = NSFont.systemFontOfSize_(12.0)
        
        # Title
        attrs = {
            NSFontAttributeName: font_medium,
            NSForegroundColorAttributeName: NSColor.whiteColor()
        }
        title = NSAttributedString.alloc().initWithString_attributes_("Legend:", attrs)
        title.drawAtPoint_(NSMakePoint(legend_x, legend_y))
        
        y_pos = legend_y + 25
        attrs[NSFontAttributeName] = font_small
        
        # Green circle - Valid points
        self.valid_dot_color.set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(legend_x + 10, y_pos + 2, 10, 10)).fill()
        text = NSAttributedString.alloc().initWithString_attributes_("Valid calibration points", attrs)
        text.drawAtPoint_(NSMakePoint(legend_x + 30, y_pos))
        y_pos += 18
        
        # Red circle - Error points
        self.error_dot_color.set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(legend_x + 10, y_pos + 2, 10, 10)).fill()
        text = NSAttributedString.alloc().initWithString_attributes_("Error points (>3° diff)", attrs)
        text.drawAtPoint_(NSMakePoint(legend_x + 30, y_pos))
        y_pos += 18
        
        # Purple diamond - Final prediction
        purple = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.8, 0.4, 1.0, 1.0)
        purple.set()
        cx, cy = legend_x + 15, y_pos + 7
        size = 6
        points = [
            NSMakePoint(cx, cy - size),
            NSMakePoint(cx + size, cy),
            NSMakePoint(cx, cy + size),
            NSMakePoint(cx - size, cy),
        ]
        path = NSBezierPath.bezierPath()
        path.moveToPoint_(points[0])
        for point in points[1:]:
            path.lineToPoint_(point)
        path.closePath()
        path.fill()
        text = NSAttributedString.alloc().initWithString_attributes_("Final prediction", attrs)
        text.drawAtPoint_(NSMakePoint(legend_x + 30, y_pos))
        y_pos += 18
        
        # Raw predictions per screen
        if self.model_params:
            raw_colors = [
                (1.0, 0.7, 0.4),  # Orange
                (0.4, 0.9, 0.9),  # Cyan
                (1.0, 1.0, 0.4),  # Yellow
                (1.0, 0.4, 0.8),  # Magenta
            ]
            raw_labels = ["Screen 0 raw", "Screen 1 raw", "Screen 2 raw", "Screen 3 raw"]
            
            for screen_idx in sorted(self.model_params.keys()):
                if screen_idx < len(raw_colors):
                    r, g, b = raw_colors[screen_idx]
                    color = NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 1.0)
                    color.set()
                    NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(legend_x + 10, y_pos + 2, 10, 10)).fill()
                    text = NSAttributedString.alloc().initWithString_attributes_(raw_labels[screen_idx], attrs)
                    text.drawAtPoint_(NSMakePoint(legend_x + 30, y_pos))
                    y_pos += 18
        
        # Add separator
        y_pos += 8
        
        # Add toggle instruction
        attrs[NSForegroundColorAttributeName] = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.8, 0.8, 0.8, 1.0)
        toggle_text = f"Press T: {'Hide' if self.show_test_points else 'Show'} test points"
        text = NSAttributedString.alloc().initWithString_attributes_(toggle_text, attrs)
        text.drawAtPoint_(NSMakePoint(legend_x, y_pos))
    
    def _draw_text(self, text, x, y, color, centered=False):
        """Draw text at position."""
        attrs = {
            NSFontAttributeName: self.font_small,
            NSForegroundColorAttributeName: color
        }
        
        attr_string = NSAttributedString.alloc().initWithString_attributes_(text, attrs)
        size = attr_string.size()
        
        if centered:
            x = x - size.width / 2
            y = y - size.height / 2
        
        point = NSMakePoint(x, y)
        attr_string.drawAtPoint_(point)


class CocoaVisualizationApp(NSObject):
    """Main visualization application using native Cocoa."""
    
    def init(self):
        """Initialize the application."""
        self = objc.super(CocoaVisualizationApp, self).init()
        if self is None:
            return None
        
        self.window = None
        self.view = None
        self.calibration_data = []
        self.data_df = None  # pandas DataFrame
        self.error_details = []
        self.model_params = {}
        self.test_data = []
        self.analysis_screen_bounds = []
        
        return self
    
    def setupVisualization_(self, csv_file):
        """Set up the visualization."""
        logging.info("=== Setting up Cocoa Calibration Visualization ===")
        
        # Load calibration data
        self._load_calibration_data(csv_file)
        
        # Perform data analysis
        self._analyze_data()
        
        # Calculate screen bounds
        self._calculate_screen_bounds()
        
        # Create visualization window
        self._create_window()
        
        logging.info("Setup complete")
        return True
    
    def _load_calibration_data(self, csv_file):
        """Load calibration data from CSV file."""
        logging.info(f"Loading calibration data from {csv_file}")
        
        try:
            # Load into list for compatibility
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    point = {
                        'screen_index': int(row['screen_index']),
                        'position_index': int(row['position_index']),
                        'cocoa_x': float(row['cocoa_x']),
                        'cocoa_y': float(row['cocoa_y']),
                        'pitch_avg': float(row['pitch_avg']),
                        'yaw_avg': float(row['yaw_avg']),
                        'sample_count': int(row['sample_count'])
                    }
                    self.calibration_data.append(point)
            
            # Also create DataFrame for analysis utilities
            # These expect 'absolute_x' and 'absolute_y' columns
            self.data_df = pd.DataFrame(self.calibration_data)
            self.data_df['absolute_x'] = self.data_df['cocoa_x']
            self.data_df['absolute_y'] = self.data_df['cocoa_y']
            
            logging.info(f"Loaded {len(self.calibration_data)} calibration points")
        
        except FileNotFoundError:
            logging.error(f"File not found: {csv_file}")
            return False
        except Exception as e:
            logging.error(f"Error loading calibration data: {e}")
            return False
        
        return True
    
    def _analyze_data(self):
        """Perform data analysis using calibration utilities."""
        logging.info("Analyzing calibration data...")
        
        # 1. Check data consistency (find error points)
        self.data_df, self.error_details = check_data_consistency(self.data_df)
        logging.info(f"Found {len(self.error_details)} consistency errors")
        
        # 2. Fit linear regression models per screen
        self.model_params = fit_linear_models(self.data_df)
        for screen_idx in sorted(self.model_params.keys()):
            params = self.model_params[screen_idx]
            logging.info(f"Screen {screen_idx}: Pitch R²={params['pitch']['r_squared']:.4f}, "
                        f"Yaw R²={params['yaw']['r_squared']:.4f}")
        
        # 3. Get screen boundaries for prediction
        self.analysis_screen_bounds = get_screen_bounds(self.data_df)
        
        # 4. Generate test data for predictions
        self.test_data = generate_test_data(self.data_df, num_samples=20)
        logging.info(f"Generated {len(self.test_data)} test predictions")
        
        # Update calibration_data with error flags from DataFrame
        for i, point in enumerate(self.calibration_data):
            point['has_error'] = bool(self.data_df.iloc[i]['has_error'])
    
    def _calculate_screen_bounds(self):
        """Calculate bounding box for each screen."""
        screen_bounds = {}
        
        for point in self.calibration_data:
            screen_idx = point['screen_index']
            x = point['cocoa_x']
            y = point['cocoa_y']
            
            if screen_idx not in screen_bounds:
                screen_bounds[screen_idx] = {
                    'min_x': x,
                    'max_x': x,
                    'min_y': y,
                    'max_y': y
                }
            else:
                screen_bounds[screen_idx]['min_x'] = min(screen_bounds[screen_idx]['min_x'], x)
                screen_bounds[screen_idx]['max_x'] = max(screen_bounds[screen_idx]['max_x'], x)
                screen_bounds[screen_idx]['min_y'] = min(screen_bounds[screen_idx]['min_y'], y)
                screen_bounds[screen_idx]['max_y'] = max(screen_bounds[screen_idx]['max_y'], y)
        
        self.screen_bounds = screen_bounds
        
        for screen_idx, bounds in screen_bounds.items():
            logging.info(f"Screen {screen_idx}: x=[{bounds['min_x']:.0f}, {bounds['max_x']:.0f}], "
                        f"y=[{bounds['min_y']:.0f}, {bounds['max_y']:.0f}]")
    
    def _create_window(self):
        """Create visualization window with fixed 1200x900 size."""
        # Fixed window size
        window_width = 1200
        window_height = 900
        
        # Center on main screen
        main_screen = NSScreen.mainScreen()
        screen_frame = main_screen.frame()
        x = screen_frame.origin.x + (screen_frame.size.width - window_width) / 2
        y = screen_frame.origin.y + (screen_frame.size.height - window_height) / 2
        
        logging.info(f"Window size: {window_width}x{window_height}")
        
        # Calculate virtual space for scaling
        all_bounds = list(self.screen_bounds.values())
        min_x = min(b['min_x'] for b in all_bounds)
        min_y = min(b['min_y'] for b in all_bounds)
        max_x = max(b['max_x'] for b in all_bounds)
        max_y = max(b['max_y'] for b in all_bounds)
        
        # Store scaling info in view
        self.world_bounds = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
        
        # Create fixed-size window
        frame = NSMakeRect(x, y, window_width, window_height)
        
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask,
            NSBackingStoreBuffered,
            False
        )
        
        self.window.setTitle_("Calibration Visualization")
        self.window.setBackgroundColor_(NSColor.blackColor())
        self.window.setOpaque_(True)
        
        # Get content rect (excludes title bar)
        content_rect = self.window.contentRectForFrameRect_(frame)
        
        # Create custom view with content rect
        self.view = VisualizationView.alloc().initWithFrame_(content_rect)
        self.view.calibration_data = self.calibration_data
        self.view.screen_bounds = self.screen_bounds
        self.view.world_bounds = self.world_bounds
        self.view.error_details = self.error_details
        self.view.model_params = self.model_params
        self.view.test_data = self.test_data
        self.view.analysis_screen_bounds = self.analysis_screen_bounds
        
        self.window.setContentView_(self.view)
        self.window.makeKeyAndOrderFront_(None)
        self.window.makeFirstResponder_(self.view)
        self.window.makeMainWindow()
        self.window.orderFrontRegardless()
        
        # Activate application
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        
        logging.info("Created visualization window")
    
    def applicationShouldTerminateAfterLastWindowClosed_(self, sender):
        """Terminate app when window is closed."""
        return True
    
    def run(self):
        """Run the visualization."""
        logging.info("Starting visualization - press ESC to exit")
        NSApplication.sharedApplication().run()


def main():
    """Main entry point."""
    import sys
    import os
    
    print("=" * 70)
    print("Native Cocoa Calibration Visualization")
    print("=" * 70)
    print()
    
    # Determine CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Look for most recent cocoa calibration file
        results_dir = "calibration_results"
        if os.path.exists(results_dir):
            files = [f for f in os.listdir(results_dir) if f.startswith("cocoa_calibration_") and f.endswith(".csv")]
            if files:
                files.sort(reverse=True)  # Most recent first
                csv_file = os.path.join(results_dir, files[0])
                print(f"Using most recent calibration: {files[0]}")
            else:
                csv_file = "cocoa_calibration.csv"
        else:
            csv_file = "cocoa_calibration.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: Calibration file not found: {csv_file}")
        print()
        print("Usage: python visualize_cocoa_calibration.py [calibration.csv]")
        print()
        print("Please run cocoa_calibration.py first to generate calibration data.")
        return
    
    print(f"Loading: {csv_file}")
    print()
    
    # Create application
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    
    # Create visualization app
    viz_app = CocoaVisualizationApp.alloc().init()
    app.setDelegate_(viz_app)
    
    # Setup
    if not viz_app.setupVisualization_(csv_file):
        logging.error("Failed to set up visualization")
        return
    
    # Run visualization
    viz_app.run()
    
    print()
    print("Visualization closed")


if __name__ == "__main__":
    main()
