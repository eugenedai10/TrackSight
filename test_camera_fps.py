#!/usr/bin/env python3
"""
Camera FPS Test Utility

Tests camera capabilities at different resolutions and frame rates.
Helps diagnose actual FPS vs requested FPS.
"""

import cv2
import time
import sys


def test_camera_config(camera_index, width, height, requested_fps):
    """Test a specific camera configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {width}x{height} @ {requested_fps} FPS")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return False
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, requested_fps)
    
    # Read back actual values
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Requested: {width}x{height} @ {requested_fps} FPS")
    print(f"Reported:  {actual_width:.0f}x{actual_height:.0f} @ {actual_fps:.0f} FPS")
    
    # Measure actual FPS by capturing frames
    print("\nMeasuring actual FPS (5 second test)...")
    frame_count = 0
    start_time = time.time()
    test_duration = 5.0
    
    while time.time() - start_time < test_duration:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
        else:
            print("WARNING: Failed to read frame")
            break
    
    elapsed = time.time() - start_time
    measured_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"Measured:  {measured_fps:.1f} FPS ({frame_count} frames in {elapsed:.1f}s)")
    
    # Check results
    if abs(measured_fps - requested_fps) < 5:
        print(f"✓ SUCCESS: Achieved {measured_fps:.1f} FPS (target: {requested_fps})")
        result = True
    else:
        print(f"✗ FAILED: Only achieved {measured_fps:.1f} FPS (target: {requested_fps})")
        result = False
    
    cap.release()
    return result


def main():
    """Main test function."""
    camera_index = 0
    
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])
    
    print("="*60)
    print("Camera FPS Test Utility")
    print("="*60)
    print(f"Testing camera index: {camera_index}")
    
    # Test configurations
    configs = [
        # (width, height, fps)
        (640, 480, 30),   # Standard 30 FPS
        (640, 480, 60),   # 60 FPS at standard resolution
        (320, 240, 60),   # 60 FPS at lower resolution
        (1280, 720, 30),  # HD at 30 FPS
        (1280, 720, 60),  # HD at 60 FPS (if supported)
    ]
    
    results = []
    for width, height, fps in configs:
        success = test_camera_config(camera_index, width, height, fps)
        results.append((width, height, fps, success))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for width, height, fps, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {width}x{height} @ {fps} FPS")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Find best 60 FPS config
    best_60fps = None
    for width, height, fps, success in results:
        if fps == 60 and success:
            best_60fps = (width, height)
            break
    
    if best_60fps:
        print(f"✓ Camera supports 60 FPS at {best_60fps[0]}x{best_60fps[1]}")
        print(f"  Use: CameraManager(camera_index={camera_index}, width={best_60fps[0]}, height={best_60fps[1]}, fps=60)")
    else:
        print("✗ Camera does not reliably support 60 FPS at tested resolutions")
        print("  Recommendation: Use 30 FPS for stable operation")
        print(f"  Use: CameraManager(camera_index={camera_index}, fps=30)")


if __name__ == "__main__":
    main()
