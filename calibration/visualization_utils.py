"""
Visualization utilities for calibration system.

Shared constants and helper functions for visualization.
"""

# Screen colors for multi-screen visualization
SCREEN_COLORS = [
    (60, 80, 120),   # Blue-ish for screen 0
    (80, 120, 80),   # Green-ish for screen 1
    (120, 80, 80),   # Red-ish for screen 2
    (120, 100, 60),  # Orange-ish for screen 3
]

# Alternative screen colors (for calibration_simple.py style)
SCREEN_COLORS_ALT = [
    (100, 100, 150),  # Blue-ish
    (150, 100, 100),  # Red-ish
    (100, 150, 100),  # Green-ish
    (150, 150, 100),  # Yellow-ish
]

# Regression line colors
LINE_COLORS = [
    (100, 130, 180),   # Blue-ish for screen 0
    (130, 180, 130),   # Green-ish for screen 1
    (180, 130, 130),   # Red-ish for screen 2
    (180, 150, 100),   # Orange-ish for screen 3
]

# Common colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 60)
LIGHT_GRAY = (200, 200, 200)
DARK_BG = (40, 40, 50)
SCREEN_BG = (30, 30, 40)

# Point colors
VALID_POINT_COLOR = (50, 255, 100)
VALID_POINT_OUTLINE = (100, 255, 150)
ERROR_POINT_COLOR = (255, 50, 50)
ERROR_POINT_OUTLINE = (255, 100, 100)

# Status colors
SUCCESS_COLOR = (0, 255, 0)
WARNING_COLOR = (255, 100, 100)
INFO_COLOR = (150, 200, 255)

# Test prediction colors
TEST_COLOR = (200, 100, 255)  # Purple for capped points
RAW_COLOR = (255, 180, 100)   # Orange for uncapped points
LINE_COLOR = (150, 150, 150)  # Gray line connecting raw to capped


def get_screen_color(screen_index, use_alt=False):
    """
    Get color for a specific screen index.
    
    Args:
        screen_index: Index of the screen
        use_alt: Whether to use alternative color scheme
        
    Returns:
        tuple: RGB color tuple
    """
    colors = SCREEN_COLORS_ALT if use_alt else SCREEN_COLORS
    return colors[screen_index % len(colors)]


def get_line_color(screen_index):
    """
    Get regression line color for a specific screen index.
    
    Args:
        screen_index: Index of the screen
        
    Returns:
        tuple: RGB color tuple
    """
    return LINE_COLORS[screen_index % len(LINE_COLORS)]


def setup_coordinate_system(data, window_width=1400, window_height=900, padding=100):
    """
    Calculate coordinate system parameters for visualization.
    
    Args:
        data: pandas DataFrame with calibration data containing 'absolute_x' and 'absolute_y'
        window_width: Width of visualization window
        window_height: Height of visualization window
        padding: Padding around the visualization
        
    Returns:
        dict: Coordinate system parameters
    """
    min_x = data['absolute_x'].min()
    max_x = data['absolute_x'].max()
    min_y = data['absolute_y'].min()
    max_y = data['absolute_y'].max()
    
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    scale_x = (window_width - 2 * padding) / world_width
    scale_y = (window_height - 2 * padding) / world_height
    scale = min(scale_x, scale_y)
    
    return {
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'padding': padding, 'scale': scale,
        'window_width': window_width,
        'window_height': window_height
    }


def world_to_screen(x, y, coord_system):
    """
    Convert world coordinates to screen coordinates.
    
    Args:
        x, y: World coordinates
        coord_system: Coordinate system dict from setup_coordinate_system()
        
    Returns:
        tuple: (screen_x, screen_y) as integers
    """
    screen_x = coord_system['padding'] + (x - coord_system['min_x']) * coord_system['scale']
    screen_y = coord_system['padding'] + (y - coord_system['min_y']) * coord_system['scale']
    return int(screen_x), int(screen_y)
