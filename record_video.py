import cv2
import argparse
import time
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Record video from default camera for gaze estimation testing")
    parser.add_argument(
        "--duration", 
        type=int, 
        required=True, 
        help="Recording duration in seconds"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Output video filename (default: recording_YYYYMMDD_HHMMSS.mp4)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30, 
        help="Frames per second for recording (default: 30)"
    )
    parser.add_argument(
        "--resolution", 
        type=str, 
        default="640x480", 
        help="Recording resolution in WIDTHxHEIGHT format (default: 640x480)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        logging.error("Invalid resolution format. Use WIDTHxHEIGHT (e.g., 640x480)")
        return
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"recording_{timestamp}.mp4"
    
    # Ensure output has .mp4 extension
    if not args.output.endswith('.mp4'):
        args.output += '.mp4'
    
    logging.info(f"Starting video recording...")
    logging.info(f"Duration: {args.duration} seconds")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Resolution: {width}x{height}")
    logging.info(f"FPS: {args.fps}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logging.error("Error: Cannot access the default camera (index 0)")
        logging.error("Please check if:")
        logging.error("1. Camera is connected and not being used by another application")
        logging.error("2. Camera permissions are granted")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Get actual camera properties (may differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logging.info(f"Camera initialized - Actual resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (actual_width, actual_height))
    
    if not out.isOpened():
        logging.error("Error: Cannot initialize video writer")
        cap.release()
        return
    
    # Recording setup
    start_time = time.time()
    frame_count = 0
    
    logging.info("\n" + "="*50)
    logging.info("RECORDING STARTED - Press 'q' to stop early")
    logging.info("="*50)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logging.error("Error: Failed to capture frame")
                break
            
            # Calculate elapsed and remaining time
            elapsed_time = time.time() - start_time
            remaining_time = max(0, args.duration - elapsed_time)
            
            # Add countdown text to frame
            countdown_text = f"Recording: {remaining_time:.1f}s remaining"
            cv2.putText(frame, countdown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Write frame to video file
            out.write(frame)
            frame_count += 1
            
            # Display frame (optional - user can see what's being recorded)
            cv2.imshow('Recording - Press q to stop', frame)
            
            # Check for early termination
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("\nRecording stopped by user")
                break
            
            # Check if duration reached
            if elapsed_time >= args.duration:
                logging.info(f"\nRecording completed - {args.duration} seconds")
                break
            
            # Progress update every second
            if int(elapsed_time) != int(elapsed_time - 1/args.fps):
                print(f"\rRecording... {remaining_time:.0f}s remaining", end='', flush=True)
    
    except KeyboardInterrupt:
        logging.info("\nRecording interrupted by user (Ctrl+C)")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        actual_duration = time.time() - start_time
        logging.info(f"\n" + "="*50)
        logging.info("RECORDING FINISHED")
        logging.info("="*50)
        logging.info(f"Output file: {args.output}")
        logging.info(f"Actual duration: {actual_duration:.2f} seconds")
        logging.info(f"Frames recorded: {frame_count}")
        logging.info(f"Average FPS: {frame_count/actual_duration:.2f}")
        
        # Check if file was created successfully
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
            logging.info(f"File size: {file_size:.2f} MB")
            logging.info(f"\nTo use with gaze estimation:")
            logging.info(f"python inference.py --model resnet50 --weight weights/resnet50.pt --source {args.output} --view --dataset gaze360")
        else:
            logging.error("Error: Output file was not created successfully")


if __name__ == "__main__":
    main()
