import numpy as np
from collections import defaultdict, deque
from .detector import ObjectDetector

class ObjectTracker:
    def __init__(self, detector, tracker_config="bytetrack.yaml", history_size=30):
        """
        Initialize the object tracker.
        
        Args:
            detector: ObjectDetector instance
            tracker_config: Configuration for the tracker
            history_size: Number of frames to keep in tracking history
        """
        self.detector = detector
        self.detector.model.tracker = tracker_config
        self.history_size = history_size
        self.track_history = defaultdict(lambda: deque(maxlen=history_size))
        self.colors = {}
        
    def track(self, frame, imgsz=640):
        """
        Track objects in the given frame.
        
        Args:
            frame: The image frame to process
            imgsz: Image size for the detector model
            
        Returns:
            YOLO results object with tracking information
        """
        results = self.detector.model.track(frame, persist=True, imgsz=imgsz)
        return results
        
    def extract_objects(self, results):
        """
        Extract tracked objects from detection results.
        
        Args:
            results: YOLO results object with tracking info
            
        Returns:
            List of tracked objects with format:
            [("tracked", class_name, track_id, x1, y1, x2, y2, confidence), ...]
        """
        objects = []
        
        if not results or len(results) == 0 or not hasattr(results[0], 'boxes'):
            return objects
        
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            confidence = boxes.conf[i].item()
            class_id = int(boxes.cls[i].item())
            class_name = self.detector.model.names[class_id]
            
            # For tracking results
            if hasattr(boxes, 'id') and boxes.id is not None:
                try:
                    track_id = int(boxes.id[i].item())
                    objects.append(("tracked", class_name, track_id, x1, y1, x2, y2, confidence))
                    
                    # Initialize color for this track_id if not exists
                    if track_id not in self.colors:
                        self.colors[track_id] = (
                            np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255)
                        )
                except:
                    # Fall back to regular detection if tracking fails
                    objects.append(("detected", class_name, None, x1, y1, x2, y2, confidence))
            else:
                # Regular detection without tracking
                objects.append(("detected", class_name, None, x1, y1, x2, y2, confidence))
        
        return objects
        
    def update_history(self, objects):
        """
        Update tracking history for objects.
        
        Args:
            objects: List of tracked objects
            
        Returns:
            Updated objects list with same format
        """
        for obj in objects:
            if obj[0] == "tracked":
                _, _, track_id, x1, y1, x2, y2, _ = obj
                
                # Calculate object center
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Update track history
                self.track_history[track_id].append((cx, cy))
                
        return objects
        
    def get_track_history(self, track_id):
        """
        Get tracking history for a specific track ID.
        
        Args:
            track_id: ID of the track to retrieve
            
        Returns:
            List of (x, y) coordinates representing the history of positions
        """
        if track_id in self.track_history:
            return list(self.track_history[track_id])
        return []
        
    def get_track_color(self, track_id):
        """
        Get color for a specific track ID.
        
        Args:
            track_id: ID of the track
            
        Returns:
            (R, G, B) color tuple for the track
        """
        if track_id not in self.colors:
            self.colors[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        return self.colors[track_id]
