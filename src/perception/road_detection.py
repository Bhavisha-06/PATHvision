import cv2
import numpy as np
import torch

class RoadDetector:
    def __init__(self, use_detectron=False):
        """
        Initialize the road detector.
        
        Args:
            use_detectron: Whether to use Detectron2 for road detection
        """
        self.use_detectron = use_detectron
        self.predictor = None
        self.road_class_id = 31  # Road class ID in COCO (street)
        
        if use_detectron:
            self.setup_detectron()
    
    def setup_detectron(self):
        """Set up road detection using Detectron2 if available."""
        try:
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.data import MetadataCatalog
            
            # Setup Detectron2 for road segmentation
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {cfg.MODEL.DEVICE}")
            
            self.predictor = DefaultPredictor(cfg)
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            print("Detectron2 model loaded successfully")
            self.use_detectron = True
        except Exception as e:
            print(f"Error loading Detectron2: {e}")
            print("Using OpenCV for road detection")
            self.use_detectron = False
    
    def detect_opencv(self, frame):
        """
        Simple road/walkable area detection using color thresholding and morphology.
        
        Args:
            frame: Input image frame
            
        Returns:
            Binary mask of detected road/walkable area
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for road colors (adjust these ranges based on your video)
        lower_gray = np.array([0, 0, 60])
        upper_gray = np.array([179, 50, 160])
        
        # Create mask for road-like areas
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Focus on the bottom half of the image where the road is most likely to be
        height = mask.shape[0]
        top_half = int(height * 0.5)
        mask[:top_half, :] = 0
        
        return mask
    
    def detect_detectron(self, frame):
        """
        Detect road using Detectron2.
        
        Args:
            frame: Input image frame
            
        Returns:
            Binary mask of detected road/walkable area
        """
        if not self.use_detectron or self.predictor is None:
            return self.detect_opencv(frame)
        
        try:
            outputs = self.predictor(frame)
            instances = outputs["instances"].to("cpu")
            
            # Create a blank mask for the walkable area
            height, width = frame.shape[:2]
            new_walkable_area_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Find road segments
            if len(instances) > 0:
                classes = instances.pred_classes.numpy()
                masks = instances.pred_masks.numpy()
                
                # Fill road segments in the mask (class 31 is 'street' in COCO)
                for i, class_id in enumerate(classes):
                    if class_id == self.road_class_id:  # Road class
                        new_walkable_area_mask = np.logical_or(new_walkable_area_mask, masks[i]).astype(np.uint8) * 255
            
            # If no road was detected, use OpenCV fallback
            if np.sum(new_walkable_area_mask) == 0:
                new_walkable_area_mask = self.detect_opencv(frame)
            
            return new_walkable_area_mask
        except Exception as e:
            print(f"Error in Detectron2 inference: {e}")
            return self.detect_opencv(frame)
    
    def detect(self, frame):
        """
        Detect road/walkable area in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Binary mask of detected road/walkable area
        """
        if self.use_detectron:
            return self.detect_detectron(frame)
        else:
            return self.detect_opencv(frame)
