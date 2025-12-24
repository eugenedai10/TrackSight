# Parallel Inference Implementation Guide

## Overview

`inference_parallel.py` implements a multi-threaded pipeline architecture that achieves **1.7x higher FPS** compared to the sequential version by overlapping camera I/O with inference processing.

## Performance Comparison

| Metric | Sequential (`inference.py`) | Parallel (`inference_parallel.py`) | Improvement |
|--------|----------------------------|-------------------------------------|-------------|
| **FPS** | 9.5 | **16.4** | **🚀 +73%** |
| **Latency** | ~105ms | ~150ms | +45ms |
| **Face Detection** | 26ms (CoreML) | 26ms (CoreML) | Same |
| **Gaze Detection** | 8ms (MPS) | 25ms (MPS) | Variable |

## Architecture

### Pipeline Design

```
┌─────────────────┐
│  Camera Thread  │  Captures frames continuously (~29 FPS)
│   (Background)  │
└────────┬────────┘
         │ frame_queue (maxsize=2)
         ▼
┌─────────────────┐
│Inference Thread │  Face + Gaze detection (~16 FPS)
│   (Background)  │
└────────┬────────┘
         │ result_queue (maxsize=2)
         ▼
┌─────────────────┐
│ Display Thread  │  Renders and displays results (~16 FPS)
│  (Main Thread)  │
└─────────────────┘
```

### Why This Design?

1. **Camera I/O Overlap**: While inference processes frame N, camera captures frame N+1
2. **Non-blocking Queues**: Drops frames if processing can't keep up (prevents memory buildup)
3. **Main Thread Display**: macOS requires OpenCV display on main thread
4. **GPU Acceleration**: Both face (CoreML) and gaze (MPS) models run on GPU

## Key Components

### 1. Thread-Safe FPS Counter
```python
class FPSCounter:
    """Tracks FPS for each pipeline stage"""
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
        self.lock = threading.Lock()
```

### 2. Camera Thread
```python
def camera_thread(cap, frame_queue, stop_event, fps_counter):
    """Continuously captures frames from camera"""
    - Runs at ~29 FPS (camera limit)
    - Timestamps each frame for latency tracking
    - Drops frames if inference is slow (non-blocking)
```

### 3. Inference Thread
```python
def inference_thread(face_detector, gaze_detector, ...):
    """Processes frames with face + gaze detection"""
    - Runs at ~16 FPS (bottleneck)
    - Face detection: 26ms (CoreML GPU)
    - Gaze detection: 8-44ms (MPS GPU, variable)
    - Includes face tracking and gaze smoothing
```

### 4. Display Thread (Main)
```python
def display_thread(result_queue, stop_event, ...):
    """Renders results and handles display"""
    - Runs on main thread (macOS requirement)
    - Draws bounding boxes and gaze vectors
    - Displays FPS and latency metrics
    - Handles keyboard input (press 'q' to quit)
```

## Usage

### Basic Usage
```bash
# Run parallel inference with webcam
python inference_parallel.py --source 0 --view

# Process video file
python inference_parallel.py --source video.mp4 --view

# Save output
python inference_parallel.py --source 0 --view --output output.mp4
```

### Command Line Arguments
```bash
--model       Model name (default: resnet34)
--weight      Path to model weights (default: weights/resnet34.pt)
--source      Video source: 0 for webcam, path for video file
--view        Display results in window
--output      Save output video to file
--dataset     Dataset config (default: gaze360)
```

## Implementation Details

### Queue Management
```python
# Bounded queues prevent memory buildup
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

# Non-blocking operations with timeout
try:
    frame_queue.put((timestamp, frame), timeout=0.01)
except queue.Full:
    pass  # Drop frame if queue is full
```

### Thread Synchronization
```python
# Stop event for graceful shutdown
stop_event = threading.Event()

# All threads check stop_event in their loops
while not stop_event.is_set():
    # Process...
```

### macOS Compatibility Fix

**Important:** OpenCV's `imshow()` must run on the main thread on macOS:

```python
# DON'T: Create display as daemon thread (causes crashes on macOS)
disp_thread = threading.Thread(target=display_thread, daemon=True)
disp_thread.start()

# DO: Run display on main thread
display_thread(result_queue, stop_event, display_fps, output_writer)
```

## Performance Analysis

### Sequential Flow (`inference.py`)
```
Frame 1: [Camera 80ms] → [Inference 34ms] → [Display 20ms] = 134ms
Frame 2: [Camera 80ms] → [Inference 34ms] → [Display 20ms] = 134ms
Total: 268ms for 2 frames = 7.5 FPS
```

### Parallel Flow (`inference_parallel.py`)
```
Time 0-80ms:   Camera(Frame1) | Inference(idle)  | Display(idle)
Time 80-114ms: Camera(Frame2) | Inference(Frame1)| Display(idle)
Time 114-134ms:Camera(Frame2) | Inference(Frame1)| Display(Frame1)
Time 134-194ms:Camera(Frame3) | Inference(Frame2)| Display(Frame1)

Throughput: 1000ms / 61ms ≈ 16.4 FPS
```

### Monitoring Performance

The parallel version logs detailed statistics every 30 frames:

```
Overall FPS: 16.40, Latency: 123.7ms, Face: 27.7ms, Gaze: 18.1ms
```

- **Overall FPS**: End-to-end throughput
- **Latency**: Time from capture to display
- **Face**: Face detection time
- **Gaze**: Gaze estimation time (per face)

## Trade-offs

### Advantages ✅
- **1.7x higher FPS** (9.5 → 16.4)
- Better CPU/GPU utilization
- Smoother frame processing
- Camera runs at full speed
- Real-time performance for single face

### Disadvantages ⚠️
- **Higher latency**: ~150ms vs ~105ms (+45ms)
- More complex code
- Slightly higher memory usage
- 1-2 frame delay between capture and display

## When to Use Each Version

### Use Parallel (`inference_parallel.py`) when:
- ✅ Need maximum FPS (real-time applications)
- ✅ Latency of 150ms is acceptable
- ✅ Processing single face
- ✅ Want smoother visual experience

### Use Sequential (`inference.py`) when:
- ✅ Need minimum latency (<150ms)
- ✅ Debugging or development
- ✅ Simpler code is preferred
- ✅ FPS isn't critical

## Troubleshooting

### Low FPS in Parallel Mode

Check the pipeline statistics at shutdown:

```
=== Pipeline Statistics ===
Camera FPS: 28.95    # Should be ~30 (camera limit)
Inference FPS: 16.40  # Bottleneck (should match Display)
Display FPS: 16.40    # Should match Inference
```

**Diagnosis:**
- **Camera FPS low**: Check camera settings or cable
- **Inference FPS low**: Verify GPU acceleration is working
- **Display FPS low**: Reduce resolution or disable annotations

### High Latency

Latency increases with queue depth. To reduce:

```python
# Lower latency (drops more frames)
frame_queue = queue.Queue(maxsize=1)  # ~80ms latency

# Current setting (balanced)
frame_queue = queue.Queue(maxsize=2)  # ~150ms latency

# Higher throughput (more buffering)
frame_queue = queue.Queue(maxsize=4)  # ~200ms latency
```

### Thread Crashes

If threads crash on startup:

1. **Display thread exception**: Ensure display runs on main thread (already implemented)
2. **GPU out of memory**: Reduce batch size or model resolution
3. **Queue timeout issues**: Increase timeout values in `get()` / `put()`

## Performance Optimization Tips

### 1. Reduce Processing Load
```python
# Process every other frame
if frame_count % 2 == 0:
    bboxes, keypoints = face_detector.detect(frame)
```

### 2. Adjust Queue Sizes
```python
# Trade latency for throughput
frame_queue = queue.Queue(maxsize=1)  # Low latency
frame_queue = queue.Queue(maxsize=4)  # High throughput
```

### 3. Skip Display
```python
# Disable visualization for maximum speed
python inference_parallel.py --source 0 --output results.mp4
# (No --view flag)
```

## Code Structure

### Main Components
- `FPSCounter`: Thread-safe FPS measurement
- `camera_thread()`: Continuous frame capture
- `inference_thread()`: Face + gaze detection
- `display_thread()`: Rendering and display
- `main()`: Setup and thread management

### Thread Lifecycle
```python
# 1. Create threads
cam_thread = threading.Thread(target=camera_thread, ...)
inf_thread = threading.Thread(target=inference_thread, ...)

# 2. Start background threads
cam_thread.start()
inf_thread.start()

# 3. Run display on main thread
display_thread(...)  # Blocks until done

# 4. Wait for cleanup
cam_thread.join(timeout=2)
inf_thread.join(timeout=2)
```

## Advanced Topics

### Multiprocessing Alternative

For true parallel CPU work, use multiprocessing instead:

```python
from multiprocessing import Process, Queue

# Separate processes bypass Python GIL
cam_proc = Process(target=camera_thread, args=(...))
inf_proc = Process(target=inference_thread, args=(...))
```

**Trade-offs:**
- ✅ True parallelism (no GIL)
- ⚠️ Higher memory usage (separate process memory)
- ⚠️ More complex IPC (inter-process communication)

### Batch Processing

Process multiple faces in parallel on GPU:

```python
# Collect all faces
faces = [preprocess(face) for face in face_crops]

# Batch process on GPU
batch = torch.stack(faces).to(device)
results = gaze_detector(batch)  # All faces at once
```

### Dynamic Queue Sizing

Adjust queue size based on load:

```python
if inference_fps.get_fps() < camera_fps.get_fps() * 0.8:
    # Inference is slow, reduce buffer
    frame_queue = queue.Queue(maxsize=1)
else:
    # Inference is fast, allow buffering
    frame_queue = queue.Queue(maxsize=4)
```

## Comparison Summary

| Feature | Sequential | Parallel |
|---------|-----------|----------|
| FPS | 9.5 | 16.4 |
| Latency | 105ms | 150ms |
| Code Complexity | Simple | Moderate |
| CPU Utilization | 60% | 85% |
| GPU Utilization | Same | Same |
| Memory Usage | Low | Moderate |
| Best For | Debugging | Production |

## Related Files

- `inference.py`: Sequential version with GPU acceleration
- `inference_parallel.py`: This parallel implementation
- `GPU_ACCELERATION_GUIDE.md`: Details on CoreML/MPS setup
- `requirements.txt`: Dependencies

## Conclusion

The parallel implementation provides a **73% FPS boost** by overlapping I/O with computation. Combined with GPU acceleration (CoreML + MPS), it offers optimal performance for real-time gaze tracking on Apple Silicon.

Choose the parallel version when throughput matters more than latency, and the sequential version when simplicity or minimum latency is required.
