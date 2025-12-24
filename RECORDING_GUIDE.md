# Video Recording Helper for Gaze Estimation

This guide explains how to use the `record_video.py` script to record test videos for gaze estimation.

## Quick Start

### Basic Recording (30 seconds)
```bash
python record_video.py --duration 30
```

### Custom Output Filename
```bash
python record_video.py --duration 60 --output my_test_video.mp4
```

### High Quality Recording
```bash
python record_video.py --duration 45 --resolution 1280x720 --fps 30 --output hd_test.mp4
```

## Command Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--duration` | Yes | - | Recording duration in seconds |
| `--output` | No | `recording_YYYYMMDD_HHMMSS.mp4` | Output video filename |
| `--fps` | No | 30 | Frames per second |
| `--resolution` | No | `640x480` | Video resolution (WIDTHxHEIGHT) |

## Usage Examples

### 1. Quick Test Recording
```bash
# Record 10 seconds for quick testing
python record_video.py --duration 10
```

### 2. Standard Test Recording
```bash
# Record 30 seconds with default settings
python record_video.py --duration 30 --output gaze_test.mp4
```

### 3. High Quality Recording
```bash
# Record 60 seconds in HD
python record_video.py --duration 60 --resolution 1280x720 --output hd_gaze_test.mp4
```

### 4. Long Duration Recording
```bash
# Record 5 minutes for comprehensive testing
python record_video.py --duration 300 --output long_test.mp4
```

## During Recording

- **Visual Feedback**: A live preview window shows what's being recorded
- **Countdown Timer**: Green text overlay shows remaining recording time
- **Early Stop**: Press 'q' in the preview window to stop recording early
- **Force Stop**: Use Ctrl+C in terminal to interrupt recording

## After Recording

The script will display:
- Output filename and location
- Actual recording duration
- Number of frames recorded
- Average FPS achieved
- File size in MB
- Command to run gaze estimation on the recorded video

## Using Recorded Videos with Gaze Estimation

After recording, use your video with the gaze estimation script:

```bash
python inference.py --model resnet50 --weight weights/resnet50.pt --source your_video.mp4 --view --dataset gaze360
```

## Tips for Best Results

### Camera Setup
- Ensure good lighting on your face
- Position camera at eye level
- Maintain 2-3 feet distance from camera
- Avoid backlighting (windows behind you)

### Recording Tips
- Look in different directions during recording
- Include natural eye movements
- Record in various lighting conditions for testing
- Keep head relatively stable for better face detection

### Recommended Durations
- **Quick test**: 10-15 seconds
- **Standard test**: 30-60 seconds  
- **Comprehensive test**: 2-5 minutes
- **Dataset creation**: 10+ minutes with varied movements

## Troubleshooting

### Camera Access Issues
```
Error: Cannot access the default camera (index 0)
```
**Solutions:**
- Close other applications using the camera
- Check camera permissions in system settings
- Try unplugging and reconnecting USB cameras
- Restart the application using the camera

### Video Writer Issues
```
Error: Cannot initialize video writer
```
**Solutions:**
- Ensure you have write permissions in the directory
- Check available disk space
- Try a different output filename
- Verify OpenCV installation includes video codecs

### Low FPS Performance
If actual FPS is much lower than requested:
- Reduce resolution (e.g., `--resolution 320x240`)
- Lower FPS setting (e.g., `--fps 15`)
- Close other resource-intensive applications
- Check camera specifications

## File Formats

- **Output Format**: MP4 (H.264 compatible)
- **Codec**: MP4V (widely supported)
- **Compatibility**: Works with OpenCV, VLC, and most video players
- **Gaze Estimation**: Fully compatible with the inference pipeline

## Integration Workflow

1. **Record test video**:
   ```bash
   python record_video.py --duration 30 --output test.mp4
   ```

2. **Run gaze estimation**:
   ```bash
   python inference.py --model resnet50 --weight weights/resnet50.pt --source test.mp4 --view --dataset gaze360
   ```

3. **Analyze results** and adjust recording parameters if needed

This creates a complete workflow for testing and validating gaze estimation performance on custom recorded content.
