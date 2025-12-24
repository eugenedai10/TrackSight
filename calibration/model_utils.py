"""
Model utilities for calibration system.

Handles model fitting and coordinate prediction from gaze angles.
"""

import numpy as np
from scipy import stats


def fit_linear_models(data):
    """
    Fit linear regression models per screen:
    - pitch = slope_pitch * x + intercept_pitch
    - yaw = slope_yaw * y + intercept_yaw
    
    Args:
        data: pandas DataFrame with calibration data
        
    Returns:
        dict: Model parameters for inverse mapping, organized by screen
    """
    model_params = {}
    
    # Fit separate models for each screen
    for screen_idx in sorted(data['screen_index'].unique()):
        screen_data = data[data['screen_index'] == screen_idx]
        
        x_vals = screen_data['absolute_x'].values
        pitch_vals = screen_data['pitch_avg'].values
        y_vals = screen_data['absolute_y'].values
        yaw_vals = screen_data['yaw_avg'].values
        
        # Fit pitch vs x
        slope_pitch, intercept_pitch, r_value_pitch, _, _ = stats.linregress(x_vals, pitch_vals)
        
        # Fit yaw vs y
        slope_yaw, intercept_yaw, r_value_yaw, _, _ = stats.linregress(y_vals, yaw_vals)
        
        model_params[screen_idx] = {
            'pitch': {'slope': slope_pitch, 'intercept': intercept_pitch, 'r_squared': r_value_pitch**2},
            'yaw': {'slope': slope_yaw, 'intercept': intercept_yaw, 'r_squared': r_value_yaw**2}
        }
    
    return model_params


def distance_to_screen_edge(x, y, screen_bound):
    """
    Calculate minimum distance from point (x, y) to screen boundary.
    
    Args:
        x, y: Point coordinates
        screen_bound: Dict with 'min_x', 'max_x', 'min_y', 'max_y'
    
    Returns:
        float: Distance to nearest edge
    """
    # Find closest point on screen rectangle by clamping to bounds
    closest_x = max(screen_bound['min_x'], min(x, screen_bound['max_x']))
    closest_y = max(screen_bound['min_y'], min(y, screen_bound['max_y']))
    
    # Calculate Euclidean distance
    distance = ((x - closest_x)**2 + (y - closest_y)**2)**0.5
    return distance


def find_screen_for_point(x, y, screen_bounds):
    """
    Determine which screen a point belongs to.
    
    Args:
        x, y: Point coordinates
        screen_bounds: List of screen boundary dicts
    
    Returns:
        dict: Screen boundary that the point belongs to
    """
    # Step 1: Check if point is inside any screen
    for screen in screen_bounds:
        if (screen['min_x'] <= x <= screen['max_x'] and 
            screen['min_y'] <= y <= screen['max_y']):
            return screen
    
    # Step 2: Find closest screen by edge distance
    min_distance = float('inf')
    closest_screen = screen_bounds[0]  # Default to first screen
    
    for screen in screen_bounds:
        dist = distance_to_screen_edge(x, y, screen)
        if dist < min_distance:
            min_distance = dist
            closest_screen = screen
    
    return closest_screen


def predict_coordinates(pitch, yaw, model_params, screen_bounds=None):
    """
    Inverse mapping: (pitch, yaw) -> (x, y, screen_index) using per-screen models.
    
    Args:
        pitch: pitch angle in degrees
        yaw: yaw angle in degrees
        model_params: dictionary with per-screen model parameters (keyed by screen_index)
        screen_bounds: optional list of screen boundary dicts for selecting screen
    
    Returns:
        tuple: (x, y, screen_index) - absolute coordinates and screen index
    """
    if screen_bounds is None:
        # If no bounds provided, use first screen's model
        screen_idx = list(model_params.keys())[0]
        params = model_params[screen_idx]
        x = (pitch - params['pitch']['intercept']) / params['pitch']['slope']
        y = (yaw - params['yaw']['intercept']) / params['yaw']['slope']
        return x, y, screen_idx
    
    # Try each screen's model and find best match
    best_screen_idx = None
    best_distance = float('inf')
    predictions = {}
    
    for screen_idx, params in model_params.items():
        # Predict using this screen's model
        x = (pitch - params['pitch']['intercept']) / params['pitch']['slope']
        y = (yaw - params['yaw']['intercept']) / params['yaw']['slope']
        predictions[screen_idx] = (x, y)
        
        # Find corresponding screen bounds
        screen_bound = next((s for s in screen_bounds if s['screen_index'] == screen_idx), None)
        if screen_bound is None:
            continue
        
        # Check if point is inside this screen
        if (screen_bound['min_x'] <= x <= screen_bound['max_x'] and 
            screen_bound['min_y'] <= y <= screen_bound['max_y']):
            # Point is inside - use this screen
            return x, y, screen_idx
        
        # Calculate distance to this screen's boundary
        dist = distance_to_screen_edge(x, y, screen_bound)
        if dist < best_distance:
            best_distance = dist
            best_screen_idx = screen_idx
    
    # No screen contained the point, use closest one
    if best_screen_idx is not None:
        x, y = predictions[best_screen_idx]
        # Clamp to screen bounds
        screen_bound = next((s for s in screen_bounds if s['screen_index'] == best_screen_idx), None)
        if screen_bound:
            x = max(screen_bound['min_x'], min(x, screen_bound['max_x']))
            y = max(screen_bound['min_y'], min(y, screen_bound['max_y']))
        return x, y, best_screen_idx
    
    # Fallback: use first screen if no best found
    screen_idx = list(model_params.keys())[0]
    x, y = predictions.get(screen_idx, (0, 0))
    return x, y, screen_idx


class CalibrationModel:
    """Encapsulates calibration models and screen bounds for gaze prediction."""
    
    def __init__(self, data):
        """
        Initialize calibration model from calibration data.
        
        Args:
            data: pandas DataFrame with calibration data
        """
        from calibration.data_utils import get_screen_bounds
        
        self.model_params = fit_linear_models(data)
        self.screen_bounds = get_screen_bounds(data)
    
    def predict(self, pitch, yaw):
        """
        Predict absolute screen coordinates from gaze angles.
        
        Args:
            pitch: pitch angle in degrees
            yaw: yaw angle in degrees
        
        Returns:
            tuple: (x, y, screen_index) - absolute coordinates and screen index
        """
        return predict_coordinates(pitch, yaw, self.model_params, self.screen_bounds)
