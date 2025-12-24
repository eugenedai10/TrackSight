# Gaze Estimation Calibration Tool - Usage Guide

## Overview

This calibration tool provides a 9-dot calibration system for gaze estimation. It displays dots sequentially on the screen and collects gaze data for each position, allowing you to create calibration mappings between screen positions and gaze angles.

## Features

- **Multi-monitor support**: Automatically detects and allows selection of display monitors
- **Precise timing**: Each dot is displayed for exactly 5 seconds with millisecond precision
- **Real-time gaze collection**: Collects gaze readings at ~30 FPS during each dot display
- **Data filtering**: Automatically filters out initial unstable readings (first 2 seconds)
- **Multiple export formats**: Exports data in both CSV and JSON formats
- **Professional interface**: Clean, distraction-free fullscreen interface

## Requirements

### Hardware
- Camera (webcam or external camera)
- One or more monitors
- Sufficient lighting for face detection

### Software Dependencies
- Python 3.8+
- pygame >= 2.0.0
- OpenCV (cv2)
- PyTorch
- numpy
- All dependencies from the main gaze estimation project

### Model Files
- `weights/resnet34.pt` - Pre-trained gaze estimation model
- `config.py` - Configuration file
- `utils/helpers.py` - Helper functions

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model files are present**:
   - Download or train the ResNet34 gaze estimation model
   - Place the model file at `weights/resnet34.pt`

3. **Test the setup**:
   ```bash
   python inference.py  # Should run without parameters
   ```

## Usage

### Basic Usage

Run the calibration application:
```bash
python calibration_app.py
```

### Calibration Process

1. **Monitor Selection**: If multiple monitors are detected, select the desired display
2. **Welcome Screen**: Read instructions and press SPACE or ENTER to start
3. **Calibration Sequence**: 
   - 9 dots will appear sequentially (top-left to bottom-right)
   - Each dot has a 3-second preparation countdown
   - Each dot is displayed for 5 seconds
   - Look directly at each dot when it appears
   - Keep your head still during the entire process
4. **Completion**: Results are automatically exported to `calibration_results/`

### Controls

- **SPACE** or **ENTER**: Start calibration / Continue
- **ESC**: Exit application at any time
- **Mouse/Arrow Keys**: Navigate monitor selection dialog

## Output Files

The calibration tool creates two output files in the `calibration_results/` directory:

### CSV File (`calibration_YYYYMMDD_HHMMSS.csv`)

Contains averaged calibration data:

| Column | Description |
|--------|-------------|
| `position_index` | Dot position (0-8) |
| `screen_x` | X coordinate on screen |
| `screen_y` | Y coordinate on screen |
| `avg_yaw_deg` | Average yaw angle in degrees |
| `avg_pitch_deg` | Average pitch angle in degrees |
| `std_yaw_deg` | Standard deviation of yaw |
| `std_pitch_deg` | Standard deviation of pitch |
| `sample_count` | Number of stable readings |
| `timestamp` | Export timestamp |

### JSON File (`calibration_raw_YYYYMMDD_HHMMSS.json`)

Contains complete raw data including:
- All individual readings with timestamps
- Calibration session metadata
- Statistical analysis for each position
- Complete position information

## Best Practices

### Before Calibration
- Ensure good lighting conditions
- Position yourself comfortably in front of the camera
- Remove or minimize distractions in the environment
- Close unnecessary applications

### During Calibration
- **Keep your head still** - This is crucial for accurate calibration
- **Look directly at each dot** - Focus on the center of each dot
- **Maintain natural posture** - Sit comfortably and naturally
- **Don't rush** - The timing is automatic, just focus on looking at the dots

### After Calibration
- Review the exported CSV file for data quality
- Check that `sample_count` values are reasonable (>50 per position)
- Verify that standard deviation values are not excessively high
- Re-run calibration if data quality is poor

## Troubleshooting

### Common Issues

**"Cannot open camera"**
- Ensure camera is connected and not in use by other applications
- Try different camera indices (0, 1, 2, etc.)
- Check camera permissions

**"Failed to initialize gaze engine"**
- Verify model file exists at `weights/resnet34.pt`
- Check that all dependencies are installed
- Ensure sufficient system resources (RAM, GPU memory)

**"No face detected"**
- Improve lighting conditions
- Position yourself within camera view
- Remove obstructions (glasses, hair, etc.)
- Ensure face is clearly visible

**Poor calibration quality**
- Redo calibration with better lighting
- Ensure head stability during calibration
- Check for camera focus issues
- Verify proper distance from screen

### Performance Optimization

**For better accuracy**:
- Use higher resolution camera
- Ensure stable lighting
- Minimize head movement
- Use larger screen for better dot separation

**For faster processing**:
- Use GPU acceleration (CUDA/MPS)
- Close unnecessary applications
- Use lower camera resolution if needed

## Advanced Usage

### Custom Configuration

You can modify calibration parameters by editing the `CalibrationApp` class:

```python
# In calibration/pygame_gui.py
self.dot_duration_ms = 5000  # Dot display time (milliseconds)
self.preparation_duration_ms = 3000  # Preparation time
self.dot_radius = 15  # Dot size in pixels
```

### Integration with Other Systems

The exported CSV data can be used with other gaze tracking systems:

```python
import pandas as pd

# Load calibration data
cal_data = pd.read_csv('calibration_results/calibration_YYYYMMDD_HHMMSS.csv')

# Create position-to-gaze mapping
position_map = {}
for _, row in cal_data.iterrows():
    position_map[(row['screen_x'], row['screen_y'])] = (row['avg_yaw_deg'], row['avg_pitch_deg'])
```

## Technical Details

### Coordinate System
- Screen coordinates: (0,0) at top-left, (width,height) at bottom-right
- Yaw angles: Negative = left, Positive = right
- Pitch angles: Negative = up, Positive = down

### Data Collection
- Sampling rate: ~30 FPS during dot display
- Filtering: First 2 seconds of each dot are discarded
- Averaging: Stable readings (seconds 2-5) are averaged per position

### Timing Precision
- Uses pygame's high-precision timing system
- Dot display duration: Exactly 5000ms ± 1ms
- Frame rate: 60 FPS for smooth display

## Support

For issues or questions:
1. Check this documentation
2. Review the troubleshooting section
3. Check the calibration log file (`calibration.log`)
4. Verify all dependencies and model files are present

## Version History

- **v1.0.0**: Initial release with 9-dot calibration, multi-monitor support, and CSV/JSON export
