import cv2
import logging
import argparse
import warnings
import numpy as np
import time

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze, GazeSmoother

import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Global smoothing configuration
GAZE_SMOOTHING_TIME = 0.1  # seconds
ENABLE_GAZE_SMOOTHING = True


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time optimized gaze estimation inference")
    parser.add_argument("--model", type=str, default="mobilenetv2", help="Model name")
    parser.add_argument(
        "--weight",
        type=str,
        default="weights/mobilenetv2.pt",
        help="Path to gaze estimation model weights"
    )
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="0",
                        help="Path to source video file or camera index")
    parser.add_argument("--output", type=str, default=None, help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
    
    # Real-time optimization parameters
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame (1=no skip, 2=every 2nd frame)")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size for gaze model (224, 320, 448)")
    parser.add_argument("--face-detect-interval", type=int, default=5, help="Run face detection every N frames")
    parser.add_argument("--max-faces", type=int, default=3, help="Maximum number of faces to track")
    parser.add_argument("--benchmark", action="store_true", help="Show FPS and timing information")
    
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def pre_process(image, input_size=224):
    """Optimized preprocessing with configurable input size."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


class FaceTracker:
    """Simple face tracker to maintain face positions between detections."""
    
    def __init__(self, max_faces=3, timeout=2.0):
        self.max_faces = max_faces
        self.timeout = timeout
        self.tracked_faces = {}  # {face_id: {'bbox': bbox, 'last_seen': time}}
        self.next_id = 0
    
    def update(self, new_bboxes, current_time):
        """Update tracked faces with new detections."""
        if len(new_bboxes) == 0:
            return []
        
        # Remove old faces
        to_remove = []
        for face_id, face_data in self.tracked_faces.items():
            if current_time - face_data['last_seen'] > self.timeout:
                to_remove.append(face_id)
        for face_id in to_remove:
            del self.tracked_faces[face_id]
        
        # Match new detections to existing faces
        matched_faces = []
        used_detections = set()
        
        for face_id, face_data in self.tracked_faces.items():
            old_bbox = face_data['bbox']
            old_center = ((old_bbox[0] + old_bbox[2]) / 2, (old_bbox[1] + old_bbox[3]) / 2)
            
            best_match = None
            best_distance = float('inf')
            
            for i, new_bbox in enumerate(new_bboxes):
                if i in used_detections:
                    continue
                    
                new_center = ((new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2)
                distance = np.sqrt((old_center[0] - new_center[0])**2 + (old_center[1] - new_center[1])**2)
                
                # Distance threshold based on face size
                face_size = max(new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1])
                threshold = face_size * 0.5
                
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Update existing face
                self.tracked_faces[face_id]['bbox'] = new_bboxes[best_match]
                self.tracked_faces[face_id]['last_seen'] = current_time
                matched_faces.append((face_id, new_bboxes[best_match]))
                used_detections.add(best_match)
        
        # Add new faces for unmatched detections
        for i, new_bbox in enumerate(new_bboxes):
            if i not in used_detections and len(self.tracked_faces) < self.max_faces:
                face_id = self.next_id
                self.next_id += 1
                self.tracked_faces[face_id] = {
                    'bbox': new_bbox,
                    'last_seen': current_time
                }
                matched_faces.append((face_id, new_bbox))
        
        return matched_faces
    
    def get_current_faces(self):
        """Get currently tracked faces."""
        return [(face_id, data['bbox']) for face_id, data in self.tracked_faces.items()]


def main(params):
    # Enhanced device detection for Apple M2 GPU support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple M2 GPU (MPS) for acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA GPU for acceleration")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for inference")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    # Initialize face detector
    face_detector = uniface.RetinaFace("retinaface_mnet025")
    face_tracker = FaceTracker(max_faces=params.max_faces)

    # Load gaze estimation model
    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device, weights_only=False)
        gaze_detector.load_state_dict(state_dict)
        logging.info(f"Gaze estimation model ({params.model}) loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load gaze estimation model: {e}")
        return

    gaze_detector.to(device)
    gaze_detector.eval()

    # Initialize video capture
    video_source = params.source
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open video source")

    # Set camera properties for better performance
    if video_source.isdigit() or video_source == '0':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize video writer if output specified
    out = None
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # Face smoothing
    face_smoothers = {}
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 30
    last_fps_update = start_time
    current_fps = 0
    
    # Timing breakdown
    face_detect_time = 0
    gaze_estimate_time = 0
    total_frames = 0
    
    logging.info("Starting real-time gaze estimation...")
    logging.info(f"Optimizations: frame_skip={params.frame_skip}, input_size={params.input_size}, face_detect_interval={params.face_detect_interval}")
    
    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            current_time = time.time()
            frame_count += 1
            total_frames += 1
            
            # Calculate video timestamp
            if video_source.isdigit() or video_source == '0':
                # For live camera, use elapsed time since start
                video_timestamp = current_time - start_time
            else:
                # For video files, use frame position and FPS
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                video_timestamp = frame_count / fps
            
            # Frame skipping optimization
            if frame_count % params.frame_skip != 0:
                if params.view:
                    cv2.imshow('Real-time Gaze Estimation', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # Face detection optimization - run less frequently
            if frame_count % params.face_detect_interval == 0:
                detect_start = time.time()
                bboxes, keypoints = face_detector.detect(frame)
                face_detect_time += time.time() - detect_start
                
                # Update face tracker
                tracked_faces = face_tracker.update(bboxes, current_time)
            else:
                # Use previously tracked faces
                tracked_faces = face_tracker.get_current_faces()

            # Process gaze estimation for tracked faces
            gaze_start = time.time()
            
            for face_id, bbox in tracked_faces:
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                
                # Ensure valid bounding box
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Extract and preprocess face
                face_image = frame[y_min:y_max, x_min:x_max]
                if face_image.size == 0:
                    continue
                    
                processed_image = pre_process(face_image, params.input_size)
                processed_image = processed_image.to(device)

                # Gaze estimation
                pitch, yaw = gaze_detector(processed_image)
                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                # Convert to angles
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                # Convert to radians
                pitch_predicted = np.radians(pitch_predicted.cpu())
                yaw_predicted = np.radians(yaw_predicted.cpu())

                # Apply smoothing
                if face_id not in face_smoothers:
                    face_smoothers[face_id] = GazeSmoother(GAZE_SMOOTHING_TIME)
                
                if ENABLE_GAZE_SMOOTHING:
                    pitch_predicted, yaw_predicted = face_smoothers[face_id].update(
                        float(pitch_predicted), float(yaw_predicted), current_time
                    )
                else:
                    pitch_predicted = float(pitch_predicted)
                    yaw_predicted = float(yaw_predicted)

                # Log yaw and pitch data with timestamp
                logging.info(f"Timestamp={video_timestamp:.3f}s, Face {face_id}: Yaw={np.degrees(yaw_predicted):.2f}°, Pitch={np.degrees(pitch_predicted):.2f}°")
                
                # Display yaw and pitch on screen
                text_x = x_min
                text_y = y_min - 10 if y_min > 30 else y_max + 20
                yaw_deg = np.degrees(yaw_predicted)
                pitch_deg = np.degrees(pitch_predicted)
                
                cv2.putText(frame, f"Yaw: {yaw_deg:.1f}°", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Pitch: {pitch_deg:.1f}°", (text_x, text_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Draw results
                draw_bbox_gaze(frame, bbox, pitch_predicted, yaw_predicted)

            gaze_estimate_time += time.time() - gaze_start

            # Calculate and display FPS
            if params.benchmark and (current_time - last_fps_update) >= 1.0:
                elapsed = current_time - last_fps_update
                current_fps = fps_update_interval / elapsed
                last_fps_update = current_time
                fps_update_interval = 30

            # Add performance info to frame
            if params.benchmark:
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Model: {params.model}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Skip: {params.frame_skip}, Size: {params.input_size}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save frame if output specified
            if out:
                out.write(frame)

            # Display frame
            if params.view:
                cv2.imshow('Real-time Gaze Estimation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Cleanup and final statistics
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Performance summary
    total_time = time.time() - start_time
    avg_fps = total_frames / total_time
    
    logging.info("\n" + "="*50)
    logging.info("PERFORMANCE SUMMARY")
    logging.info("="*50)
    logging.info(f"Total frames processed: {total_frames}")
    logging.info(f"Total time: {total_time:.2f} seconds")
    logging.info(f"Average FPS: {avg_fps:.2f}")
    logging.info(f"Face detection time: {face_detect_time:.2f}s ({face_detect_time/total_time*100:.1f}%)")
    logging.info(f"Gaze estimation time: {gaze_estimate_time:.2f}s ({gaze_estimate_time/total_time*100:.1f}%)")
    logging.info(f"Model: {params.model}")
    logging.info(f"Input size: {params.input_size}px")
    logging.info(f"Frame skip: {params.frame_skip}")
    logging.info(f"Face detect interval: {params.face_detect_interval}")


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --output must be provided.")

    main(args)
