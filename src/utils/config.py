"""
Configuration management for the Vehicle Perception System.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any


@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    model_path: str = 'yolov9s.pt'
    confidence_threshold: float = 0.5
    imgsz: int = 640
    device: Optional[str] = None  # None will auto-select


@dataclass
class TrackingConfig:
    """Configuration for object tracking."""
    enabled: bool = True
    tracker_type: str = "bytetrack.yaml"
    history_size: int = 30  # Number of frames to keep in tracking history


@dataclass
class CollisionConfig:
    """Configuration for collision detection and prediction."""
    time_to_collision_threshold: float = 3.0  # Time threshold in seconds
    danger_threshold: float = 0.6  # Threshold for considering an object dangerous
    obstacle_classes: List[str] = None
    
    def __post_init__(self):
        if self.obstacle_classes is None:
            self.obstacle_classes = ['person', 'bicycle', 'car', 'motorcycle', 
                                     'bus', 'truck', 'dog', 'cow', 'elephant', 
                                     'bear', 'horse', 'sheep']


@dataclass
class RoadDetectionConfig:
    """Configuration for road/walkable area detection."""
    method: str = 'opencv'  # 'opencv' or 'detectron'
    hsv_lower_bound: List[int] = None
    hsv_upper_bound: List[int] = None
    update_frequency: int = 5  # Update every N frames
    
    def __post_init__(self):
        if self.hsv_lower_bound is None:
            self.hsv_lower_bound = [0, 0, 60]
        if self.hsv_upper_bound is None:
            self.hsv_upper_bound = [179, 50, 160]


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    show_tracks: bool = True
    show_road: bool = True
    show_collision_prediction: bool = True
    show_labels: bool = True
    panel_height: int = 80  # Height of info panel in pixels


@dataclass
class SystemConfig:
    """Main configuration for the Vehicle Perception System."""
    detection: DetectionConfig = None
    tracking: TrackingConfig = None
    collision: CollisionConfig = None
    road_detection: RoadDetectionConfig = None
    visualization: VisualizationConfig = None
    output_dir: str = './data/output'
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.tracking is None:
            self.tracking = TrackingConfig()
        if self.collision is None:
            self.collision = CollisionConfig()
        if self.road_detection is None:
            self.road_detection = RoadDetectionConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)


def load_config(config_file: str) -> SystemConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to the configuration YAML file
        
    Returns:
        SystemConfig object with loaded configuration
    """
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Create detection config
        detection_dict = config_dict.get('detection', {})
        detection_config = DetectionConfig(
            model_path=detection_dict.get('model_path', 'yolov9s.pt'),
            confidence_threshold=detection_dict.get('confidence_threshold', 0.5),
            imgsz=detection_dict.get('imgsz', 640),
            device=detection_dict.get('device')
        )
        
        # Create tracking config
        tracking_dict = config_dict.get('tracking', {})
        tracking_config = TrackingConfig(
            enabled=tracking_dict.get('enabled', True),
            tracker_type=tracking_dict.get('tracker_type', 'bytetrack.yaml'),
            history_size=tracking_dict.get('history_size', 30)
        )
        
        # Create collision config
        collision_dict = config_dict.get('collision', {})
        collision_config = CollisionConfig(
            time_to_collision_threshold=collision_dict.get('time_to_collision_threshold', 3.0),
            danger_threshold=collision_dict.get('danger_threshold', 0.6),
            obstacle_classes=collision_dict.get('obstacle_classes')
        )
        
        # Create road detection config
        road_dict = config_dict.get('road_detection', {})
        road_config = RoadDetectionConfig(
            method=road_dict.get('method', 'opencv'),
            hsv_lower_bound=road_dict.get('hsv_lower_bound'),
            hsv_upper_bound=road_dict.get('hsv_upper_bound'),
            update_frequency=road_dict.get('update_frequency', 5)
        )
        
        # Create visualization config
        viz_dict = config_dict.get('visualization', {})
        viz_config = VisualizationConfig(
            show_tracks=viz_dict.get('show_tracks', True),
            show_road=viz_dict.get('show_road', True),
            show_collision_prediction=viz_dict.get('show_collision_prediction', True),
            show_labels=viz_dict.get('show_labels', True),
            panel_height=viz_dict.get('panel_height', 80)
        )
        
        # Create system config
        return SystemConfig(
            detection=detection_config,
            tracking=tracking_config,
            collision=collision_config,
            road_detection=road_config,
            visualization=viz_config,
            output_dir=config_dict.get('output_dir', './data/output')
        )
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
        return SystemConfig()


def save_config(config: SystemConfig, config_file: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: SystemConfig object to save
        config_file: Path to save the configuration YAML file
    """
    # Convert the config to a dictionary
    config_dict = {
        'detection': {
            'model_path': config.detection.model_path,
            'confidence_threshold': config.detection.confidence_threshold,
            'imgsz': config.detection.imgsz,
            'device': config.detection.device
        },
        'tracking': {
            'enabled': config.tracking.enabled,
            'tracker_type': config.tracking.tracker_type,
            'history_size': config.tracking.history_size
        },
        'collision': {
            'time_to_collision_threshold': config.collision.time_to_collision_threshold,
            'danger_threshold': config.collision.danger_threshold,
            'obstacle_classes': config.collision.obstacle_classes
        },
        'road_detection': {
            'method': config.road_detection.method,
            'hsv_lower_bound': config.road_detection.hsv_lower_bound,
            'hsv_upper_bound': config.road_detection.hsv_upper_bound,
            'update_frequency': config.road_detection.update_frequency
        },
        'visualization': {
            'show_tracks': config.visualization.show_tracks,
            'show_road': config.visualization.show_road,
            'show_collision_prediction': config.visualization.show_collision_prediction,
            'show_labels': config.visualization.show_labels,
            'panel_height': config.visualization.panel_height
        },
        'output_dir': config.output_dir
    }
    # Add this to utils/config.py

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            self.config = get_default_config()
    
    def save(self, config_path: str):
        save_config(self.config, config_path)

    def get_config(self) -> SystemConfig:
        return self.config


def get_config_value(key: str, default: Any = None) -> Any:
    # Fallback function if you're accessing a global config dictionary
    try:
        config = load_config("path/to/default.yaml")  # Replace with actual path
        return getattr(config, key, default)
    except Exception as e:
        print(f"Could not get config value for {key}: {e}")
        return default


    
    # Save to YAML file
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            print(f"Configuration saved to {config_file}")
    except Exception as e:
        print(f"Error saving configuration: {e}")


def get_default_config() -> SystemConfig:
    """Get the default configuration."""
    return SystemConfig()
