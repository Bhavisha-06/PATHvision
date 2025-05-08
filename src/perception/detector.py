import torch
import ssl
from ultralytics import YOLO

# Fix for SSL certificate verification error on macOS
ssl._create_default_https_context = ssl._create_unverified_context


class ObjectDetector:
    def __init__(self, model_path='yolov9s.pt'):
        """Initialize the object detector with a YOLO model."""
        # Load YOLO model for object detection
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect(self, frame, imgsz=640):
        """
        Detect objects in the given frame.
        
        Args:
            frame: The image frame to process
            imgsz: Image size for the YOLO model
            
        Returns:
            YOLO results object
        """
        results = self.model(frame, imgsz=imgsz)
        return results
        
    def extract_objects(self, results):
        """
        Extract objects from detection results.
        
        Args:
            results: YOLO results object
            
        Returns:
            List of detected objects with format:
            [("detected", class_name, None, x1, y1, x2, y2, confidence), ...]
        """
        objects = []
        
        if not results or len(results) == 0 or not hasattr(results[0], 'boxes'):
            return objects
        
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            confidence = boxes.conf[i].item()
            class_id = int(boxes.cls[i].item())
            class_name = self.model.names[class_id]
            
            # Regular detection without tracking
            objects.append(("detected", class_name, None, x1, y1, x2, y2, confidence))
        
        return objects
