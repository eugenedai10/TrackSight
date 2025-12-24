# Multi-Screen Automatic Calibration Implementation Plan

## Overview
Implemented automatic multi-screen calibration system for gaze estimation that uses macOS NSWindow API to programmatically move the pygame window across multiple screens.

## Implementation Date
December 18, 2025

## Problem Statement
The original calibration system could not calibrate multiple screens because:
1. Pygame windows can only draw within their own viewport
2. Cannot span window across multiple physical screens
3. Manual window dragging between screens is tedious and error-prone

## Solution Architecture

### Core Concept
**Sequential Per-Screen Calibration**
- Detect all screens via macOS NSScreen API
- Automatically move window to each screen
- Calibrate 4 corner points per screen
- Track which screen each calibration point belongs to

### Key Components

#### 1. Screen Detection (`detect_screen_configuration()`)
```python
from AppKit import NSScreen
```
- Detects all connected displays
- Retrieves origin coordinates and dimensions
- Stores configuration for automatic window positioning

#### 2. Automatic Window Movement (`move_window_to_screen()`)
```python
from AppKit import NSApplication
import Cocoa
```
- Gets pygame window handle via NSApplication
- Uses `setFrame_display_animate_()` to move/resize window
- Converts between coordinate systems:
  - macOS: bottom-left origin
  - Standard: top-left origin

#### 3. Sequential Calibration Loop
```python
for screen_idx in range(num_screens):
    show_screen_transition(screen_idx)
    move_window_to_screen(screen_idx)
    run_calibration_sequence(screen_idx)
```

#### 4. Enhanced Result Tracking
CSV format includes:
```csv
screen_index,position_index,relative_x,relative_y,window_x,window_y,absolute_x,absolute_y,
timestamp,pitch_min,pitch_max,pitch_avg,yaw_min,yaw_max,yaw_avg,sample_count
```

## Calibration Flow

### For Single Screen
1. Show setup screen (resize/position window)
2. Show welcome screen
3. Calibrate 4 corners
4. Save results

### For Multiple Screens
1. Auto-detect screens
2. Show welcome screen
3. **For each screen:**
   - Show transition countdown (3 seconds)
   - Move window to screen automatically
   - Resize to match screen dimensions
   - Calibrate 4 corners
   - Store screen_index with each point
4. Save comprehensive results

## User Experience

### Transition Screen
```
Moving to Screen 2 of 3
Window will automatically move
Starting in 3...
```

### Calibration Screen
```
Dot 1/4
[White dot with countdown timer]
Pitch: -12.3° Yaw: 45.6°
```

### Completion
```
Calibration Complete!
Displayed 8 dots (2 screens × 4 points)
Results saved to: calibration_results/simple_calibration_20251218_220015.csv
```

## Technical Details

### Coordinate Systems

#### Window-Relative Coordinates
```python
(inset, inset)  # Top-left corner
```
- Origin: top-left of window
- Range: 0 to window dimensions

#### Absolute Screen Coordinates
```python
absolute_x = relative_x + window_x
absolute_y = relative_y + window_y
```
- Origin: top-left of primary screen
- Accounts for multi-screen arrangements

### Screen Detection Example
```
Screen 0: origin=(-429, 0), size=1728x2557  # Vertical monitor (left)
Screen 1: origin=(1299, 0), size=3440x1415  # Ultrawide (center)
```

### Coordinate Transformation
```python
# macOS uses bottom-left origin
main_screen_height = NSScreen.mainScreen().frame().size.height
macos_y = main_screen_height - screen_y - screen_height

# Create frame
frame = NSMakeRect(screen_x, macos_y, width, height)
```

## Benefits

### Automated Process
- ✅ No manual window dragging
- ✅ Consistent calibration across screens
- ✅ Eliminates user positioning errors
- ✅ Faster calibration workflow

### Comprehensive Data
- ✅ Screen index tracked for each point
- ✅ Both relative and absolute coordinates
- ✅ Window position recorded
- ✅ Ready for multi-screen gaze mapping

### User-Friendly
- ✅ Clear countdown timers
- ✅ Progress indicators
- ✅ Automatic screen transitions
- ✅ Visual feedback during calibration

## CSV Output Structure

```csv
screen_index,position_index,relative_x,relative_y,window_x,window_y,absolute_x,absolute_y,timestamp,pitch_avg,yaw_avg,...
0,0,15,15,0,0,15,15,2025-12-18T22:00:15,-28.9,-67.5,...
0,1,1713,15,0,0,1713,15,2025-12-18T22:00:20,-28.8,67.2,...
1,0,15,15,1728,0,1743,15,2025-12-18T22:00:45,-29.1,-45.3,...
1,1,3425,15,1728,0,5153,15,2025-12-18T22:00:50,-28.7,44.8,...
```

## Future Enhancements

### Potential Improvements
1. **User Control**
   - Pause/resume between screens
   - Repeat calibration for specific screen
   - Manual screen selection

2. **Advanced Calibration**
   - Variable dot count per screen
   - Junction point calibration at screen edges
   - Adaptive calibration based on screen type

3. **Validation**
   - Post-calibration accuracy test
   - Heat map visualization
   - Cross-screen consistency checks

4. **Integration**
   - Real gaze inference (replace simulation)
   - Model training with multi-screen data
   - Screen-specific correction coefficients

## Testing Recommendations

### Single Screen
```bash
python calibration_simple.py
# Should work exactly as before (4 points)
```

### Dual Screen
```bash
python calibration_simple.py
# Should auto-detect 2 screens
# Move window to screen 1, calibrate 4 points
# Move window to screen 2, calibrate 4 points
# Total: 8 calibration points
```

### Triple+ Screens
```bash
python calibration_simple.py
# Sequential calibration of N screens
# Total: N × 4 points
```

## Dependencies

### Required
- `pygame` - Window management and rendering
- `pyobjc-framework-Cocoa` - macOS NSWindow/NSScreen APIs
- `numpy` - Statistical calculations

### Platform
- macOS only (uses AppKit/Cocoa)
- For other platforms, would need alternative window APIs

## Code Organization

### Main Classes
- `SimpleCalibrationApp` - Main calibration application

### Key Methods
- `detect_screen_configuration()` - Detect screens
- `move_window_to_screen()` - Window positioning
- `show_screen_transition()` - Transition UI
- `run_calibration_sequence()` - Per-screen calibration
- `run()` - Main loop with multi-screen support

### Configuration
```python
dot_duration_ms = 5000        # 5 seconds per dot
preparation_duration_ms = 3000  # 3 second countdown
transition_duration_ms = 3000   # 3 second between screens
```

## Success Criteria

- [x] Detect multiple screens on macOS
- [x] Automatically move window to each screen
- [x] Resize window to match screen dimensions
- [x] Calibrate 4 corners per screen
- [x] Track screen_index for each point
- [x] Save comprehensive CSV results
- [x] Smooth transitions with countdowns
- [x] Proper coordinate transformation
- [x] Error handling and logging
- [ ] Test with actual 2+ screen setup
- [ ] Integrate real gaze inference

## Conclusion

This implementation provides a robust, automated multi-screen calibration system that:
1. Eliminates manual window positioning
2. Ensures consistent calibration across all screens
3. Tracks screen-specific data for accurate multi-screen gaze mapping
4. Provides excellent user experience with clear feedback

The system is ready for testing on multi-screen setups and integration with real gaze estimation models.
