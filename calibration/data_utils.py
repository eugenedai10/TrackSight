"""
Data utilities for calibration system.

Handles loading, validation, and analysis of calibration data.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def load_calibration_data(filename='simple_calibration.csv'):
    """
    Load and prepare calibration data from CSV file.
    Supports both pygame format (absolute_x/y) and Cocoa format (cocoa_x/y).
    
    Args:
        filename: Path to calibration CSV file
        
    Returns:
        pandas DataFrame with calibration data
    """
    df = pd.read_csv(filename)
    
    # Handle column name differences between pygame and Cocoa calibration
    if 'cocoa_x' in df.columns and 'cocoa_y' in df.columns:
        # Cocoa format - rename columns to standard names
        df = df.rename(columns={'cocoa_x': 'absolute_x', 'cocoa_y': 'absolute_y'})
    
    # Extract needed columns
    data = df[['screen_index', 'absolute_x', 'absolute_y', 'pitch_avg', 'yaw_avg']].copy()
    return data


def check_data_consistency(data, threshold=3.0):
    """
    Check for consistency errors in calibration data.
    
    Points with matching x-coordinates should have similar pitch values,
    and points with matching y-coordinates should have similar yaw values.
    
    Args:
        data: pandas DataFrame with calibration data
        threshold: Maximum acceptable difference in degrees
        
    Returns:
        tuple: (data with 'has_error' column added, list of error details)
    """
    error_flags = [False] * len(data)
    error_details = []
    
    # Check pitch consistency for matching absolute_x
    x_groups = defaultdict(list)
    for idx, row in data.iterrows():
        x_groups[row['absolute_x']].append({'index': idx, 'pitch': row['pitch_avg']})
    
    for x_val, points in x_groups.items():
        if len(points) > 1:
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    p1, p2 = points[i], points[j]
                    diff_abs = abs(p1['pitch'] - p2['pitch'])
                    if diff_abs > threshold:
                        error_flags[p1['index']] = True
                        error_flags[p2['index']] = True
                        error_details.append({
                            'type': 'pitch',
                            'points': [p1['index'], p2['index']],
                            'x': x_val,
                            'diff_abs': diff_abs
                        })
    
    # Check yaw consistency for matching absolute_y
    y_groups = defaultdict(list)
    for idx, row in data.iterrows():
        y_groups[row['absolute_y']].append({'index': idx, 'yaw': row['yaw_avg']})
    
    for y_val, points in y_groups.items():
        if len(points) > 1:
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    p1, p2 = points[i], points[j]
                    diff_abs = abs(p1['yaw'] - p2['yaw'])
                    if diff_abs > threshold:
                        error_flags[p1['index']] = True
                        error_flags[p2['index']] = True
                        error_details.append({
                            'type': 'yaw',
                            'points': [p1['index'], p2['index']],
                            'y': y_val,
                            'diff_abs': diff_abs
                        })
    
    data['has_error'] = error_flags
    return data, error_details


def get_screen_bounds(data):
    """
    Extract boundary information for each screen from calibration data.
    
    Args:
        data: pandas DataFrame with calibration data
        
    Returns:
        list: List of dicts with screen boundary information
    """
    screen_bounds = []
    for screen_idx in sorted(data['screen_index'].unique()):
        screen_data = data[data['screen_index'] == screen_idx]
        screen_bounds.append({
            'screen_index': screen_idx,
            'min_x': screen_data['absolute_x'].min(),
            'max_x': screen_data['absolute_x'].max(),
            'min_y': screen_data['absolute_y'].min(),
            'max_y': screen_data['absolute_y'].max()
        })
    return screen_bounds


def generate_test_data(data, num_samples=10, margin=0.1):
    """
    Generate random test (pitch, yaw) pairs within calibration range.
    
    Args:
        data: pandas DataFrame with calibration data
        num_samples: Number of test samples to generate
        margin: Extra margin beyond min/max values (as fraction of range)
        
    Returns:
        list: List of (pitch, yaw) tuples
    """
    pitch_min = data['pitch_avg'].min()
    pitch_max = data['pitch_avg'].max()
    yaw_min = data['yaw_avg'].min()
    yaw_max = data['yaw_avg'].max()
    
    # Add margin
    pitch_margin = (pitch_max - pitch_min) * margin
    yaw_margin = (yaw_max - yaw_min) * margin
    
    # Generate random test points
    np.random.seed(42)  # For reproducibility
    test_pitches = np.random.uniform(pitch_min - pitch_margin, pitch_max + pitch_margin, num_samples)
    test_yaws = np.random.uniform(yaw_min - yaw_margin, yaw_max + yaw_margin, num_samples)
    
    test_data = list(zip(test_pitches, test_yaws))
    return test_data
