# Gaze Mouse Control Application Usage Guide

## Overview

The Gaze Mouse Control application allows you to control your mouse cursor using eye gaze tracking. It uses the calibration data from your 9-point calibration to accurately map where you're looking to cursor position on the screen.

## Prerequisites

### 1. Hardware Requirements
- Webcam or external camera
- macOS system
- Sufficient lighting for face detection

### 2. Software Requirements
- Python 3.8+
- All dependencies installed: `pip install -r requirements.txt`
- Gaze estimation model weights in `weights/` directory
- Calibration data file (default: `calibration.csv`)

### 3. macOS Accessibility Permissions
**IMPORTANT**: The application requires accessibility permissions to control the mouse.

#### Setting up Permissions:
1. Run the application once - it will automatically open System Preferences
2. Go to: **System Preferences → Security & Privacy → Privacy → Accessibility**
3. Click the lock icon and enter your admin password
4. Add **Python** (or **Terminal** if running from terminal) to the list
5. Check the box to enable permissions
6. Restart the application

## Running the Application

### Basic Usage
```bash
python gaze_mouse_control.py
```

### With Custom Options
```bash
# Use specific calibration file
python gaze_mouse_control.py --calibration my_calibration.csv

# Use different camera
python gaze_mouse_control.py --camera 1

# Combined options
python gaze_mouse_control.py --calibration calibration.csv --camera 0
```

## Application Controls

### Keyboard Controls
- **SPACE**: Toggle gaze control ON/OFF
- **P**: Toggle camera preview window
- **Q** or **ESC**: Quit application
- **Emergency Stop**: Move mouse to top-left corner of screen

### Visual Feedback
The camera preview window shows:
- **Face Detection Status**: Green = detected, Red = not detected
- **Gaze Control Status**: ON/OFF indicator
- **Current Gaze Angles**: Real-time pitch and yaw values
- **Control Instructions**: Keyboard shortcuts

## How It Works

### 1. Calibration Data Loading
- Loads your 9-point calibration data from CSV file
- Maps screen coordinates to gaze angles (pitch/yaw)
- Creates interpolation functions for smooth cursor movement

### 2. Real-time Gaze Tracking
- Captures video from camera
- Detects face using RetinaFace
- Estimates gaze direction using ResNet34 model
- Applies smoothing to reduce jitter

### 3. Coordinate Transformation
- Converts gaze angles to screen coordinates
- Uses bilinear interpolation between calibration points
- Handles extrapolation for areas outside calibration bounds

### 4. Mouse Control
- Moves cursor to calculated screen position
- Applies boundary constraints to keep cursor on screen
- Uses PyAutoGUI for cross-platform mouse control

## Performance Optimization

### Expected Performance
- **Frame Rate**: 20-30 FPS typical
- **Latency**: 50-100ms from eye movement to cursor movement
- **Accuracy**: Depends on calibration quality and lighting conditions

### Improving Performance
1. **Better Lighting**: Ensure good, even lighting on your face
2. **Camera Position**: Position camera at eye level, 18-24 inches away
3. **Stable Head Position**: Keep head relatively still during use
4. **Quality Calibration**: Perform calibration carefully for best accuracy

## Troubleshooting

### Common Issues

#### 1. "Accessibility permissions required" Error
- **Solution**: Follow the permission setup steps above
- **Note**: You may need to restart the application after granting permissions

#### 2. Camera Not Found
- **Check**: Camera is connected and not used by other applications
- **Try**: Different camera index: `--camera 1` or `--camera 2`
- **Verify**: Camera works in other applications

#### 3. Face Not Detected
- **Lighting**: Ensure adequate lighting on your face
- **Position**: Sit directly in front of camera
- **Distance**: Stay 18-24 inches from camera
- **Background**: Avoid cluttered backgrounds

#### 4. Inaccurate Cursor Movement
- **Recalibrate**: Run calibration again with better conditions
- **Check Data**: Verify calibration.csv has 9 valid data points
- **Lighting**: Ensure consistent lighting between calibration and use

#### 5. Jittery Cursor Movement
- **Normal**: Some jitter is expected due to natural eye movements
- **Smoothing**: The application includes built-in smoothing
- **Head Movement**: Try to keep head more stable

### Performance Issues

#### Low Frame Rate
- **GPU**: Ensure using GPU acceleration (MPS/CUDA)
- **Camera**: Lower camera resolution if needed
- **Background**: Close other resource-intensive applications

#### High CPU Usage
- **Model**: Consider using lighter face detection model
- **Preview**: Disable preview window with 'P' key
- **Resolution**: Reduce camera resolution

## Advanced Configuration

### Modifying Smoothing
Edit `gaze_mouse_control.py` and adjust:
```python
self.gaze_smoother = GazeSmoother(0.1)  # Increase for more smoothing
```

### Changing Camera Settings
Modify camera properties in `initialize_camera()`:
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Width
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height
self.cap.set(cv2.CAP_PROP_FPS, 30)            # Frame rate
```

## Safety Features

### Built-in Safety
- **Failsafe**: PyAutoGUI failsafe - move mouse to corner to stop
- **Boundary Checking**: Cursor constrained to screen bounds
- **Toggle Control**: Easy on/off with spacebar
- **Emergency Stop**: Multiple ways to stop the application

### Best Practices
- **Start Disabled**: Application starts with gaze control enabled - use spacebar to disable initially
- **Test First**: Test in safe environment before important work
- **Have Backup**: Keep regular mouse/trackpad available
- **Take Breaks**: Eye tracking can be tiring - take regular breaks

## File Structure

```
gaze-estimation/
├── gaze_mouse_control.py      # Main application
├── calibration.csv            # Your calibration data
├── weights/
│   └── resnet34.pt           # Gaze estimation model
├── requirements.txt          # Dependencies
└── GAZE_MOUSE_USAGE.md      # This guide
```

## Logging and Debugging

### Log Information
The application provides detailed logging:
- Calibration data loading status
- Device selection (CPU/GPU)
- Real-time performance statistics
- Error messages and warnings

### Debug Mode
For more detailed output, modify logging level:
```python
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
```

## Support and Troubleshooting

### Getting Help
1. Check this usage guide first
2. Verify all prerequisites are met
3. Check log output for error messages
4. Ensure calibration data is valid

### Common Log Messages
- `"Accessibility permissions OK"` - Permissions working correctly
- `"Face: Not Found"` - Improve lighting or camera position
- `"Interpolation failed"` - Check calibration data quality
- `"Using Apple M2 GPU (MPS)"` - GPU acceleration active

## Tips for Best Results

### Calibration Tips
- Perform calibration in same lighting as usage
- Keep head position consistent
- Look directly at each calibration dot
- Ensure all 9 points are captured accurately

### Usage Tips
- Start with gaze control disabled (spacebar to toggle)
- Practice in low-stakes environment first
- Use preview window to monitor face detection
- Take breaks to avoid eye strain
- Keep backup input method available

### Environment Setup
- Consistent lighting (avoid shadows on face)
- Stable seating position
- Camera at eye level
- Minimal background distractions
- Quiet environment for concentration

---

**Note**: This is an experimental application. While functional, it may not be suitable for all use cases. Always have alternative input methods available.
