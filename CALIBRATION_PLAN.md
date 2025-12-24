# Gaze Estimation Calibration App - Implementation Plan

## Project Overview
- **Goal**: Create a gaze estimation calibration tool with 9-dot display
- **Framework**: Pygame (for precise timing and multi-monitor support)
- **Integration**: Uses existing inference.py gaze estimation pipeline
- **Output**: CSV file with position-to-gaze mapping data

## Requirements Summary
- Display 9 dots in 3×3 grid (top-left to bottom-right sequence)
- Each dot displays for exactly 5 seconds
- Multi-monitor support with user selection
- Fullscreen calibration interface matching system resolution
- Real-time gaze data collection and averaging
- CSV export with timestamp

## Technical Specifications
- **Default inference.py parameters**: 
  - Model: resnet34
  - Weight: weights/resnet34.pt
  - Source: 0 (camera)
  - View: True
  - Dataset: gaze360
- **Data collection strategy**: Skip first 2 seconds per dot, average seconds 2-5
- **Timing precision**: Pygame clock system for exact 5-second intervals
- **Screen adaptation**: Full resolution of selected monitor
- **Framework choice**: Pygame selected over tkinter for superior timing and fullscreen support

## File Structure
```
gaze-estimation/
├── calibration_app.py          # Main entry point
├── calibration/                # Calibration module
│   ├── __init__.py
│   ├── monitor_utils.py        # Multi-monitor detection and selection
│   ├── gaze_engine.py          # Gaze estimation integration
│   ├── calibration_data.py     # Data collection and storage
│   └── pygame_gui.py           # Main GUI implementation
├── calibration_results/        # Output directory
│   └── calibration_YYYYMMDD_HHMMSS.csv
└── CALIBRATION_PLAN.md         # This plan document
```

## Implementation Steps (12 Phases)

### Phase 1: Foundation Setup
**Step 1: Update inference.py defaults**
- Modify `parse_args()` function default values:
  - `model`: "mobilenetv2" → "resnet34"
  - `weight`: "weights/mobilenetv2.pt" → "weights/resnet34.pt"
  - `source`: "assets/in_video.mp4" → "0"
- Test that `python inference.py` works without parameters

**Step 2: Create project directory structure**
```bash
mkdir calibration
mkdir calibration_results
touch calibration/__init__.py
touch calibration/monitor_utils.py
touch calibration/gaze_engine.py
touch calibration/calibration_data.py
touch calibration/pygame_gui.py
touch calibration_app.py
```

### Phase 2: Core Components
**Step 3: Implement monitor utilities**
- Create `MonitorUtils` class in `monitor_utils.py`
- Implement monitor detection using pygame
- Add monitor selection dialog functionality
- Test multi-monitor detection

**Step 4: Create gaze engine integration**
- Extract gaze estimation logic from `inference.py`
- Create `GazeEngine` class in `gaze_engine.py`
- Implement model loading and camera setup
- Add single-frame gaze reading method
- Test gaze readings work independently

**Step 5: Implement data management**
- Create `CalibrationData` class in `calibration_data.py`
- Add methods for storing position/gaze pairs
- Implement averaging and statistics calculation
- Add CSV export functionality
- Test data collection and export

### Phase 3: GUI Implementation
**Step 6: Basic pygame framework**
- Create main `CalibrationApp` class in `pygame_gui.py`
- Implement pygame initialization and basic window
- Add monitor selection interface
- Test fullscreen mode on selected monitor

**Step 7: Calibration sequence core**
- Implement 9-dot position calculation (3×3 grid)
- Add dot rendering and display logic
- Create precise 5-second timing system
- Add progress tracking and display

**Step 8: Integration and data collection**
- Connect gaze engine to GUI main loop
- Implement real-time gaze data collection during dot display
- Add data filtering (ignore first 2 seconds per dot)
- Store collected data using CalibrationData class

### Phase 4: User Experience
**Step 9: Enhanced interface**
- Add countdown timer before each dot
- Implement progress indicators
- Add instruction text and user guidance
- Create smooth transitions between dots

**Step 10: Results and export**
- Display calibration completion message
- Show summary statistics (accuracy per position)
- Implement automatic CSV export with timestamp
- Add option to view/save results

### Phase 5: Testing and Polish
**Step 11: Comprehensive testing**
- Test on single and multiple monitor setups
- Verify timing accuracy (exactly 5 seconds per dot)
- Test gaze data collection and averaging
- Validate CSV export format and data

**Step 12: Error handling and robustness**
- Add camera connection error handling
- Handle model loading failures gracefully
- Add user cancellation support (ESC key)
- Implement proper cleanup on exit

## Key Implementation Details

### Dot Position Calculation
```python
def _calculate_dot_positions(self, screen_width, screen_height):
    """Calculate 9 dot positions in 3x3 grid"""
    margin = 0.1  # 10% margin from edges
    positions = []
    
    for row in range(3):
        for col in range(3):
            x = margin + col * (1 - 2*margin) / 2
            y = margin + row * (1 - 2*margin) / 2
            positions.append((int(x * screen_width), int(y * screen_height)))
    
    return positions
```

### Calibration Sequence Flow
```python
def calibration_sequence(self):
    """Main calibration loop"""
    for i, position in enumerate(self.dot_positions):
        self.show_countdown(3)  # 3-second preparation
        self.show_dot(position, duration=5000)  # 5 seconds
        self.collect_gaze_data(position)
        self.update_progress(i + 1, len(self.dot_positions))
    
    self.process_results()
    self.show_results_dialog()
```

### Data Collection Strategy
- Collect gaze readings at 30+ FPS during each 5-second window
- Discard first 1-2 seconds to allow eye movement stabilization
- Average readings from seconds 2-5 for each position
- Store raw data and calculated averages

### Output Format (CSV)
```csv
position_index,screen_x,screen_y,avg_yaw_deg,avg_pitch_deg,std_yaw_deg,std_pitch_deg,sample_count,timestamp
0,192,108,-15.2,8.7,2.1,1.8,90,2025-12-16T11:30:15
1,960,108,-2.1,9.1,1.9,2.0,89,2025-12-16T11:30:25
...
```

## Key Milestones
- ✅ **Milestone 1**: inference.py runs with no parameters
- ✅ **Milestone 2**: Monitor detection and selection works
- ✅ **Milestone 3**: Gaze readings work in isolation
- ✅ **Milestone 4**: Basic fullscreen dot display works
- ✅ **Milestone 5**: Complete calibration sequence with data collection
- ✅ **Milestone 6**: CSV export with proper data format

## Implementation Status: COMPLETED ✅

All planned features have been successfully implemented and tested:

### ✅ Completed Features
- Multi-monitor detection and selection using pygame
- Gaze engine integration with existing inference.py pipeline
- 9-dot calibration sequence with precise 5-second timing
- Real-time gaze data collection with threading
- Data filtering (skip first 2 seconds per dot)
- CSV and JSON export with comprehensive data
- Professional pygame-based GUI with fullscreen support
- Comprehensive error handling and user feedback
- Complete documentation and usage guide

### 📁 Created Files
- `calibration_app.py` - Main application entry point
- `calibration/__init__.py` - Module initialization
- `calibration/monitor_utils.py` - Multi-monitor support
- `calibration/gaze_engine.py` - Gaze estimation integration
- `calibration/calibration_data.py` - Data management and export
- `calibration/pygame_gui.py` - Main GUI implementation
- `CALIBRATION_USAGE.md` - Comprehensive usage documentation
- Updated `requirements.txt` with pygame dependency
- Updated `inference.py` with new default parameters

### 🧪 Testing Results
- ✅ Monitor detection: Successfully detected 2 monitors
- ✅ Data management: Successfully exported CSV and JSON files
- ✅ Dependency checks: All required packages available
- ✅ Model file validation: All required files present
- ✅ End-to-end calibration: Complete workflow functional

### 📊 Output Format
CSV files contain position-to-gaze mappings with statistics:
```csv
position_index,screen_x,screen_y,avg_yaw_deg,avg_pitch_deg,std_yaw_deg,std_pitch_deg,sample_count,timestamp
0,192,108,-15.2,8.7,2.1,1.8,90,2025-12-16T11:30:15
```

JSON files contain complete raw data and metadata for detailed analysis.

## Dependencies
- pygame (for GUI and timing)
- opencv-python (already in project)
- torch (already in project)
- numpy (already in project)
- All existing project dependencies

## Testing Strategy
- Unit test each component individually
- Integration testing after each phase
- Multi-monitor configuration testing
- Timing accuracy validation
- Data export format verification

---
*Plan created: 2025-12-16*
*Framework: Pygame-based implementation*
*Integration: Existing gaze estimation pipeline*
