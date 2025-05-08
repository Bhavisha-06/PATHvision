"""
Main Vehicle Perception System class that integrates all components
"""

import os
import ssl
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch

from utils.config import get_config_value
from .detector import ObjectDetector
from .road_detection import RoadDetector
from .collision import CollisionPredictor
from utils.visualization import Visualizer


class VehiclePerceptionSystem:
    """
    A comprehensive vehicle perception system that integrates object detection,
    road area detection, collision prediction, and visualization.
    """
    
    def __init__(self, config=None):
        """
        Initialize the vehicle perception system.
        
        Args:
            config: Configuration dictionary or path to config file
        """
        # Fix for SSL certificate verification error on macOS
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Load configuration
        self.config = config or {}
        
        # Determine device
        device = get_config_value(self.config, "system.device", "auto")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components
        self._init_components()
        
        # Track history for objects
        self.track_history = defaultdict(lambda: deque(
            maxlen=get_config_value(self.config, "tracking.history_size", 30)
        ))
        
        # Visualization colors for tracking
        self.colors = {}
        
        print(f"Vehicle Perception System initialized on device: {self.device}")
    
    def _init_components(self):
        """Initialize all system components."""
        # Initialize object detector
        self.detector = ObjectDetector(
            model_path=get_config_value(self.config, "detection.model", "yolov9s.pt"),
            conf_thresh=get_config_value(self.config, "detection.confidence_threshold", 0.25),
            iou_thresh=get_config_value(self.config, "detection.iou_threshold", 0.45),
            img_size=get_config_value(self.config, "detection.img_size", 640),
            device=self.device,
            use_tracking=get_config_value(self.config, "tracking.enabled", True),
            tracker_type=get_config_value(self.config, "tracking.tracker_type", "bytetrack.yaml")
        )
        
        # Initialize road detector
        self.road_detector = RoadDetector(
            use_detectron=get_config_value(self.config, "road_detection.use_detectron", "auto"),
            device=self.device,
            hsv_lower=get_config_value(self.config, "road_detection.hsv_lower", [0, 0, 60]),
            hsv_upper=get_config_value(self.config, "road_detection.hsv_upper", [179, 50, 160])
        )
        
        # Initialize collision predictor
        self.collision_predictor = CollisionPredictor(
            ttc_threshold=get_config_value(self.config, "collision.ttc_threshold", 3.0),
            danger_threshold=get_config_value(self.config, "collision.danger_threshold", 0.6),
            min_history_points=get_config_value(self.config, "collision.min_history_points", 5),
            obstacle_classes=get_config_value(
                self.config, 
                "classes.obstacle_classes", 
                ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
                 'dog', 'cow', 'elephant', 'bear', 'horse', 'sheep']
            )
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer(
            config=self.config
        )
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Processed frame with visualizations
        """
        # Detect objects
        detection_results = self.detector.detect(frame)
        
        # Extract objects
        objects = self.detector.extract_objects(detection_results)
        
        # Detect road/walkable area
        walkable_area_mask = self.road_detector.detect(frame)
        
        # Update tracking history for objects
        if self.detector.use_tracking:
            self._update_tracking_history(objects)
        
        # Determine vehicle action using collision prediction
        action, confidence, warning, collision_objects = (
            self.collision_predictor.determine_action(
                objects, walkable_area_mask, frame.shape, self.track_history
            )
        )
        
        # Visualize results
        result_frame = self.visualizer.visualize(
            frame, objects, walkable_area_mask, 
            action, warning, collision_objects,
            self.track_history, self.colors
        )
        
        return result_frame
    
    def _update_tracking_history(self, objects):
        """Update tracking history for objects."""
        for obj in objects:
            if obj[0] == "tracked":
                _, class_name, track_id, x1, y1, x2, y2, conf = obj
                
                # Calculate object center
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Update track history
                self.track_history[track_id].append((cx, cy))
                
                # Initialize color for this track_id if not exists
                if track_id not in self.colors:
                    self.colors[track_id] = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255)
                    )
    
    def process_video(self, input_path, output_path=None, process_every_n_frames=None):
        """
        Process a video file with the vehicle perception system.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file (created if not exists)
            process_every_n_frames: Process road detection every N frames
        """
        # Use config value if not specified
        if process_every_n_frames is None:
            process_every_n_frames = get_config_value(
                self.config, "road_detection.frame_skip", 5
            )
        
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video file: {input_path}")
            return False
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory if it doesn't exist
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process variables
        frame_count = 0
        walkable_area_mask = None
        
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Tracking enabled: {self.detector.use_tracking}")
        
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            result_frame = self.process_frame(frame)
            
            # Write frame to output video
            out.write(result_frame)
            
            # Display progress
            if frame_count % 50 == 0 or frame_count == total_frames:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
                
                print(f"Processed {frame_count}/{total_frames} frames " +
                      f"({frame_count/total_frames*100:.1f}%) - " +
                      f"FPS: {fps_processing:.2f} - ETA: {eta:.1f}s")
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"Processing complete! {frame_count} frames processed in {elapsed:.2f}s")
        print(f"Output saved to {output_path}")
        
        return True
    
    def process_camera(self, camera_id=0, display=True, output_path=None):
        """
        Process video from a camera.
        
        Args:
            camera_id: Camera ID (default: 0 for webcam)
            display: Whether to display the processed video
            output_path: Path to output video file (optional)
        """
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error opening camera with ID: {camera_id}")
            return False
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Estimated FPS for camera
        
        # Initialize video writer if output path is specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        print(f"Processing camera feed from camera ID: {camera_id}")
        print(f"Video dimensions: {frame_width}x{frame_height}")
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result_frame = self.process_frame(frame)
            
            # Write frame to output video if enabled
            if out:
                out.write(result_frame)
            
            # Display result if enabled
            if display:
                cv2.imshow("Vehicle Perception System", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Calculate FPS every second
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processing at {fps_processing:.2f} FPS")
        
        # Release resources
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        return True
