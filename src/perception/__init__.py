"""
Core perception modules for the Vehicle Perception System.

This package contains modules for:
- Object detection
- Object tracking
- Road/walkable area detection
- Collision prediction and risk assessment
"""

from .detector import ObjectDetector
from .tracker import ObjectTracker
from .road_detection import RoadDetector
from .collision import CollisionPredictor

__all__ = [
    'ObjectDetector',
    'ObjectTracker',
    'RoadDetector',
    'CollisionPredictor'
]
