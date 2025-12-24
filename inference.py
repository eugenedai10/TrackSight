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
import onnxruntime as ort

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Global smoothing configuration
GAZE_SMOOTHING_TIME = 0.1  # seconds
ENABLE_GAZE_SMOOTHING = True


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
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

# - `retinaface_mnet025` (MobileNet v1 0.25x - smallest/fastest)

# - `retinaface_mnet050` (MobileNet v1 0.50x)

# - `retinaface_mnet_v1` (MobileNet v1)

# - `retinaface_mnet_v2` (MobileNet v2)

# - `retinaface_r18` (ResNet-18)

# - `retinaface_r34` (ResNet-34)


    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device, weights_only=False)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)
    gaze_detector.eval()

    video_source = params.source
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Face tracking for smoothing
    face_smoothers = {}  # Dictionary to track smoothers for each face
    face_timeout = 2.0  # Remove face smoothers after 2 seconds of no detection
    
    # Video timing
    start_time = time.time()
    frame_count = 0
    
    with torch.no_grad():
        while True:
            success, frame = cap.read()

            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            current_time = time.time()
            frame_count += 1
            
            # Calculate video timestamp
            if video_source.isdigit() or video_source == '0':
                # For live camera, use elapsed time since start
                video_timestamp = current_time - start_time
            else:
                # For video files, use frame position and FPS
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                video_timestamp = frame_count / video_fps
            
            # Calculate actual processing FPS
            elapsed_time = current_time - start_time
            processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Display FPS on frame (top-left corner)
            cv2.putText(frame, f"FPS: {processing_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Log FPS and timing breakdown periodically (every 30 frames)
            if frame_count % 30 == 0:
                logging.info(f"FPS: {processing_fps:.2f}, Face: {face_detect_time*1000:.1f}ms, Gaze: {gaze_detect_total*1000:.1f}ms")
            
            # Measure face detection time
            face_detect_start = time.time()
            bboxes, keypoints = face_detector.detect(frame)
            face_detect_time = time.time() - face_detect_start
            
            # Clean up old face smoothers
            faces_to_remove = []
            for face_id, smoother in face_smoothers.items():
                if smoother.last_update_time is not None and (current_time - smoother.last_update_time) > face_timeout:
                    faces_to_remove.append(face_id)
            for face_id in faces_to_remove:
                del face_smoothers[face_id]
            
            current_face_centers = []
            
            gaze_detect_total = 0
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                image = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image)
                image = image.to(device)

                gaze_detect_start = time.time()
                pitch, yaw = gaze_detector(image)
                gaze_detect_total += time.time() - gaze_detect_start

                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                # Degrees to Radians
                pitch_predicted = np.radians(pitch_predicted.cpu())
                yaw_predicted = np.radians(yaw_predicted.cpu())

                # Face association based on bounding box center
                face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                current_face_centers.append(face_center)
                
                # Find closest existing face smoother or create new one
                face_id = None
                min_distance = float('inf')
                face_size = max(x_max - x_min, y_max - y_min)
                distance_threshold = face_size * 0.5  # 50% of face size
                
                for existing_face_id, smoother in face_smoothers.items():
                    if existing_face_id not in [fc for fc in current_face_centers[:-1]]:  # Not already matched
                        # Calculate distance between face centers
                        existing_center = existing_face_id
                        distance = np.sqrt((face_center[0] - existing_center[0])**2 + 
                                         (face_center[1] - existing_center[1])**2)
                        if distance < min_distance and distance < distance_threshold:
                            min_distance = distance
                            face_id = existing_face_id

                if face_id is None:
                    # Create new smoother for this face
                    face_id = face_center
                    face_smoothers[face_id] = GazeSmoother(GAZE_SMOOTHING_TIME)

                # Apply smoothing if enabled
                if ENABLE_GAZE_SMOOTHING:
                    pitch_predicted, yaw_predicted = face_smoothers[face_id].update(
                        float(pitch_predicted), float(yaw_predicted), current_time
                    )
                else:
                    pitch_predicted = float(pitch_predicted)
                    yaw_predicted = float(yaw_predicted)

                # Log yaw and pitch data with timestamp
                yaw_deg = np.degrees(yaw_predicted)
                pitch_deg = np.degrees(pitch_predicted)
                # logging.info(f"Timestamp={video_timestamp:.3f}s, Yaw={yaw_deg:.2f}°, Pitch={pitch_deg:.2f}°")
                
                # Display yaw and pitch on screen
                text_x = x_min
                text_y = y_min - 10 if y_min > 30 else y_max + 20
                
                cv2.putText(frame, f"Yaw: {yaw_deg:.1f}°", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Pitch: {pitch_deg:.1f}°", (text_x, text_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # draw box and gaze direction
                draw_bbox_gaze(frame, bbox, pitch_predicted, yaw_predicted)

            if params.output:
                out.write(frame)

            if params.view:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --ouput must be provided.")

    main(args)
