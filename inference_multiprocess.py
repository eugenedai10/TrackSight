import cv2
import logging
import argparse
import warnings
import numpy as np
import time
import os
from multiprocessing import Process, Queue, Event
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
    parser = argparse.ArgumentParser(description="Multiprocess gaze estimation inference")
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
    """Preprocess face image for gaze detection"""
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


def camera_process(source, frame_queue, stop_event):
    """
    Camera capture process - runs independently
    Captures frames and sends to inference process via queue
    """
    logging.info("Camera process started (PID: %d)", os.getpid())
    
    # Initialize camera in this process
    if source.isdigit() or source == '0':
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logging.error("Camera process: Failed to open video source")
        stop_event.set()
        return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Camera process: Failed to read frame")
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # Log camera FPS periodically
            if frame_count % 100 == 0:
                elapsed = timestamp - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logging.info(f"Camera FPS: {fps:.2f}")
            
            try:
                # Non-blocking put with timeout
                frame_queue.put({
                    'timestamp': timestamp,
                    'frame': frame,
                    'frame_id': frame_count
                }, timeout=0.01)
            except:
                # Drop frame if queue is full
                pass
    
    except KeyboardInterrupt:
        logging.info("Camera process interrupted")
    finally:
        cap.release()
        logging.info("Camera process stopped")


def inference_process(params, frame_queue, result_queue, stop_event):
    """
    Inference process - runs independently with own GPU context
    Performs face detection and gaze estimation
    """
    logging.info("Inference process started (PID: %d)", os.getpid())
    
    # Initialize device in this process
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Inference process: Using Apple M2 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Inference process: Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logging.info("Inference process: Using CPU")
    
    # Initialize models in this process
    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)
    
    # Monkey-patch uniface for CoreML
    original_init_model = uniface.RetinaFace._initialize_model
    
    def patched_init_model(self, model_path: str) -> None:
        try:
            available_providers = ort.get_available_providers()
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if 'CoreMLExecutionProvider' in available_providers else ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            
            used_provider = self.session.get_providers()[0]
            if used_provider == 'CoreMLExecutionProvider':
                logging.info(f"Inference process: Face detector using CoreML (GPU)")
            else:
                logging.info(f"Inference process: Face detector using {used_provider}")
        except Exception as e:
            logging.error(f"Failed to load face detector: {e}")
            raise RuntimeError(f"Failed to initialize face detector") from e
    
    uniface.RetinaFace._initialize_model = patched_init_model
    face_detector = uniface.RetinaFace("retinaface_mnet025")
    
    # Load gaze detector
    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device, weights_only=False)
        gaze_detector.load_state_dict(state_dict)
        gaze_detector.to(device)
        gaze_detector.eval()
        logging.info("Inference process: Gaze detector loaded")
    except Exception as e:
        logging.error(f"Failed to load gaze detector: {e}")
        stop_event.set()
        return
    
    # Face smoothers (process-local)
    face_smoothers = {}
    frame_count = 0
    start_time = time.time()
    
    try:
        with torch.no_grad():
            while not stop_event.is_set():
                try:
                    frame_data = frame_queue.get(timeout=0.1)
                except:
                    continue
                
                timestamp = frame_data['timestamp']
                frame = frame_data['frame']
                frame_id = frame_data['frame_id']
                
                frame_count += 1
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
                    
                    # Face tracking
                    face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
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
                
                # Log inference FPS periodically
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logging.info(f"Inference FPS: {fps:.2f}, Face: {face_time*1000:.1f}ms, Gaze: {gaze_time*1000:.1f}ms")
                
                try:
                    result_queue.put({
                        'timestamp': timestamp,
                        'frame': frame,
                        'results': results,
                        'face_time': face_time,
                        'gaze_time': gaze_time,
                        'inference_time': total_inference,
                        'frame_id': frame_id
                    }, timeout=0.01)
                except:
                    # Drop if display is slow
                    pass
    
    except KeyboardInterrupt:
        logging.info("Inference process interrupted")
    finally:
        logging.info("Inference process stopped")


def display_main(result_queue, stop_event, output_path=None):
    """
    Display function - runs on main process for macOS compatibility
    Renders results and handles window display
    """
    logging.info("Display started on main process")
    
    # Setup output writer if specified
    output_writer = None
    if output_path:
        # Will be initialized when first frame arrives
        pass
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            try:
                result_data = result_queue.get(timeout=0.1)
            except:
                continue
            
            timestamp = result_data['timestamp']
            frame = result_data['frame']
            results = result_data['results']
            face_time = result_data['face_time']
            gaze_time = result_data['gaze_time']
            
            frame_count += 1
            current_time = time.time()
            
            # Calculate overall FPS
            elapsed = current_time - start_time
            overall_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Calculate end-to-end latency
            latency = (current_time - timestamp) * 1000  # ms
            
            # Initialize output writer on first frame
            if output_writer is None and output_path:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
            
            # Draw results
            for bbox, pitch, yaw in results:
                draw_bbox_gaze(frame, bbox, pitch, yaw)
                
                # Display angles
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
            cv2.putText(frame, f"[MULTIPROCESS]", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Log stats periodically
            if frame_count % 30 == 0:
                logging.info(f"Overall FPS: {overall_fps:.2f}, Latency: {latency:.1f}ms, "
                            f"Face: {face_time*1000:.1f}ms, Gaze: {gaze_time*1000:.1f}ms")
            
            # Write to output
            if output_writer is not None:
                output_writer.write(frame)
            
            # Display
            cv2.imshow('Multiprocess Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    
    except KeyboardInterrupt:
        logging.info("Display interrupted")
    finally:
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()
        logging.info("Display stopped")


def main(params):
    import os
    
    logging.info("\n=== Starting Multiprocess Pipeline ===")
    logging.info("Camera Process -> Inference Process -> Display (Main)")
    logging.info("Press 'q' to quit\n")
    
    # Create multiprocessing queues
    frame_queue = Queue(maxsize=2)
    result_queue = Queue(maxsize=2)
    
    # Shared stop event
    stop_event = Event()
    
    # Create processes
    cam_proc = Process(
        target=camera_process,
        args=(params.source, frame_queue, stop_event),
        daemon=True
    )
    
    inf_proc = Process(
        target=inference_process,
        args=(params, frame_queue, result_queue, stop_event),
        daemon=True
    )
    
    # Start background processes
    cam_proc.start()
    logging.info(f"Camera process started: PID {cam_proc.pid}")
    
    inf_proc.start()
    logging.info(f"Inference process started: PID {inf_proc.pid}")
    
    # Run display on main process (required for macOS)
    try:
        output_path = params.output if params.output else None
        display_main(result_queue, stop_event, output_path)
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
        stop_event.set()
    
    # Wait for processes to finish
    logging.info("Waiting for processes to stop...")
    cam_proc.join(timeout=2)
    inf_proc.join(timeout=2)
    
    # Force terminate if still alive
    if cam_proc.is_alive():
        logging.warning("Camera process did not stop gracefully, terminating...")
        cam_proc.terminate()
        cam_proc.join(timeout=1)
    
    if inf_proc.is_alive():
        logging.warning("Inference process did not stop gracefully, terminating...")
        inf_proc.terminate()
        inf_proc.join(timeout=1)
    
    logging.info("\n=== Pipeline Statistics ===")
    logging.info("All processes stopped cleanly")


if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    import os
    
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --output must be provided.")

    main(args)
