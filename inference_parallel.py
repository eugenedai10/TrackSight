import cv2
import logging
import argparse
import warnings
import numpy as np
import time
import threading
import queue
from collections import deque

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze, GazeSmoother

import uniface
import onnxruntime as ort

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Global smoothing configuration
GAZE_SMOOTHING_TIME = 0.1  # seconds
ENABLE_GAZE_SMOOTHING = True


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet34`")
    parser.add_argument(
        "--weight",
        type=str,
        default="weights/resnet34.pt",
        help="Path to gaze esimation model weights"
    )
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="0",
                        help="Path to source video file or camera index")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
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


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


class FPSCounter:
    """Thread-safe FPS counter"""
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def update(self):
        with self.lock:
            self.times.append(time.time())
    
    def get_fps(self):
        with self.lock:
            if len(self.times) < 2:
                return 0.0
            elapsed = self.times[-1] - self.times[0]
            if elapsed == 0:
                return 0.0
            return (len(self.times) - 1) / elapsed


def camera_thread(cap, frame_queue, stop_event, fps_counter):
    """Thread for camera capture"""
    logging.info("Camera thread started")
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Camera thread: Failed to read frame")
            break
        
        fps_counter.update()
        
        try:
            # Non-blocking put with timeout
            frame_queue.put((time.time(), frame), timeout=0.01)
        except queue.Full:
            # Drop frame if inference is slow
            pass
    
    logging.info("Camera thread stopped")


def inference_thread(face_detector, gaze_detector, device, idx_tensor, params,
                    frame_queue, result_queue, stop_event, fps_counter, face_smoothers):
    """Thread for face and gaze detection"""
    logging.info("Inference thread started")
    
    with torch.no_grad():
        while not stop_event.is_set():
            try:
                timestamp, frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Measure inference time
            inference_start = time.time()
            
            # Face detection
            face_start = time.time()
            bboxes, keypoints = face_detector.detect(frame)
            face_time = time.time() - face_start
            
            # Gaze detection
            gaze_start = time.time()
            results = []
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                
                face_image = frame[y_min:y_max, x_min:x_max]
                if face_image.size == 0:
                    continue
                    
                face_image = pre_process(face_image)
                face_image = face_image.to(device)
                
                pitch, yaw = gaze_detector(face_image)
                
                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)
                
                # Mapping from binned to angles
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                
                # Degrees to Radians
                pitch_predicted = np.radians(pitch_predicted.cpu())
                yaw_predicted = np.radians(yaw_predicted.cpu())
                
                # Face tracking for smoothing
                face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                
                # Simple face tracking by center position
                face_id = face_center
                if face_id not in face_smoothers:
                    face_smoothers[face_id] = GazeSmoother(GAZE_SMOOTHING_TIME)
                
                # Apply smoothing
                if ENABLE_GAZE_SMOOTHING:
                    pitch_predicted, yaw_predicted = face_smoothers[face_id].update(
                        float(pitch_predicted), float(yaw_predicted), time.time()
                    )
                else:
                    pitch_predicted = float(pitch_predicted)
                    yaw_predicted = float(yaw_predicted)
                
                results.append((bbox, pitch_predicted, yaw_predicted))
            
            gaze_time = time.time() - gaze_start
            total_inference = time.time() - inference_start
            
            fps_counter.update()
            
            try:
                result_queue.put((timestamp, frame, results, face_time, gaze_time, total_inference), timeout=0.01)
            except queue.Full:
                # Drop if display is slow
                pass
    
    logging.info("Inference thread stopped")


def display_thread(result_queue, stop_event, fps_counter, output_writer):
    """Thread for rendering and display"""
    logging.info("Display thread started")
    
    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        try:
            timestamp, frame, results, face_time, gaze_time, inference_time = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        frame_count += 1
        current_time = time.time()
        
        # Calculate overall FPS
        elapsed = current_time - start_time
        overall_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Calculate end-to-end latency
        latency = (current_time - timestamp) * 1000  # ms
        
        # Draw results
        for bbox, pitch, yaw in results:
            draw_bbox_gaze(frame, bbox, pitch, yaw)
            
            # Display angles on screen
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            text_x = x_min
            text_y = y_min - 10 if y_min > 30 else y_max + 20
            
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            
            cv2.putText(frame, f"Yaw: {yaw_deg:.1f}°", (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Pitch: {pitch_deg:.1f}°", (text_x, text_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display FPS and stats
        cv2.putText(frame, f"FPS: {overall_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {latency:.0f}ms", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        fps_counter.update()
        
        # Log stats periodically
        if frame_count % 30 == 0:
            logging.info(f"Overall FPS: {overall_fps:.2f}, Latency: {latency:.1f}ms, "
                        f"Face: {face_time*1000:.1f}ms, Gaze: {gaze_time*1000:.1f}ms")
        
        # Write to output if specified
        if output_writer is not None:
            output_writer.write(frame)
        
        # Display
        cv2.imshow('Parallel Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
    
    logging.info("Display thread stopped")


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

    # Monkey-patch uniface to enable CoreML acceleration for face detection
    original_init_model = uniface.RetinaFace._initialize_model
    
    def patched_init_model(self, model_path: str) -> None:
        """Modified initialization to use CoreML execution provider for Apple Silicon."""
        try:
            # Try CoreML first (uses Apple GPU and Neural Engine)
            available_providers = ort.get_available_providers()
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if 'CoreMLExecutionProvider' in available_providers else ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            
            used_provider = self.session.get_providers()[0]
            if used_provider == 'CoreMLExecutionProvider':
                logging.info(f"Face detector using CoreML (Apple GPU/Neural Engine)")
            else:
                logging.info(f"Face detector using {used_provider}")
        except Exception as e:
            logging.error(f"Failed to load model from '{model_path}': {e}")
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e
    
    # Apply the patch
    uniface.RetinaFace._initialize_model = patched_init_model
    
    # Initialize face detector with CoreML acceleration
    face_detector = uniface.RetinaFace("retinaface_mnet025")

    # Load gaze detector
    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device, weights_only=False)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)
    gaze_detector.eval()

    # Open video source
    video_source = params.source
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Setup output writer if specified
    output_writer = None
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    # Create queues for inter-thread communication
    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    
    # Thread control
    stop_event = threading.Event()
    
    # FPS counters for each stage
    camera_fps = FPSCounter()
    inference_fps = FPSCounter()
    display_fps = FPSCounter()
    
    # Face smoothers (shared between threads - will use GIL for safety)
    face_smoothers = {}

    logging.info("\n=== Starting Parallel Pipeline ===")
    logging.info("Camera -> Inference -> Display")
    logging.info("Press 'q' to quit\n")

    # Create and start background threads (camera and inference)
    cam_thread = threading.Thread(
        target=camera_thread,
        args=(cap, frame_queue, stop_event, camera_fps),
        daemon=True
    )
    
    inf_thread = threading.Thread(
        target=inference_thread,
        args=(face_detector, gaze_detector, device, idx_tensor, params,
              frame_queue, result_queue, stop_event, inference_fps, face_smoothers),
        daemon=True
    )

    # Start background threads
    cam_thread.start()
    inf_thread.start()

    # Run display on main thread (required for macOS)
    try:
        display_thread(result_queue, stop_event, display_fps, output_writer)
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
        stop_event.set()

    # Wait for background threads to finish
    logging.info("Waiting for threads to stop...")
    cam_thread.join(timeout=2)
    inf_thread.join(timeout=2)

    # Cleanup
    cap.release()
    if output_writer is not None:
        output_writer.release()
    cv2.destroyAllWindows()
    
    logging.info("\n=== Pipeline Statistics ===")
    logging.info(f"Camera FPS: {camera_fps.get_fps():.2f}")
    logging.info(f"Inference FPS: {inference_fps.get_fps():.2f}")
    logging.info(f"Display FPS: {display_fps.get_fps():.2f}")
    logging.info("Pipeline stopped cleanly")


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --output must be provided.")

    main(args)
