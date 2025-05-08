"""
Visualization utilities for the Vehicle Perception System.
Provides functions for visualizing detection results, tracks, 
collision predictions, and system status.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def draw_bounding_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a bounding box on the frame.
    
    Args:
        frame: Input frame
        x1, y1, x2, y2: Bounding box coordinates
        color: RGB color for the bounding box
        thickness: Line thickness
        
    Returns:
        Frame with bounding box
    """
    return cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(
    frame: np.ndarray,
    x: int, y: int,
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    bg_color: Optional[Tuple[int, int, int]] = None,
    font_scale: float = 0.5,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a text label on the frame.
    
    Args:
        frame: Input frame
        x, y: Position to draw the label
        label: Text to display
        color: RGB color for the text
        bg_color: RGB color for background (None for no background)
        font_scale: Font scale factor
        thickness: Line thickness
        
    Returns:
        Frame with label
    """
    # Calculate label width and height
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Draw background if specified
    if bg_color is not None:
        cv2.rectangle(
            frame,
            (x, y - label_height - baseline - 5),
            (x + label_width, y),
            bg_color,
            -1
        )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x, y - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    return frame


def draw_trajectory(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    max_history: int = 30
) -> np.ndarray:
    """
    Draw a trajectory on the frame from a list of points.
    
    Args:
        frame: Input frame
        points: List of (x, y) points
        color: RGB color for the trajectory
        max_history: Maximum history size for calculating line thickness
        
    Returns:
        Frame with trajectory
    """
    for i in range(1, len(points)):
        # Calculate thickness (thicker for more recent points)
        thickness = int(np.sqrt(max_history / float(i + 1)) * 2)
        
        # Draw line segment
        cv2.line(
            frame,
            (int(points[i-1][0]), int(points[i-1][1])),
            (int(points[i][0]), int(points[i][1])),
            color,
            thickness
        )
    
    return frame


def draw_collision_prediction(
    frame: np.ndarray,
    current_point: Tuple[int, int],
    predicted_points: List[Tuple[float, float]],
    time_to_collision: float,
    color: Tuple[int, int, int] = (0, 0, 255),
    circle_radius: int = 25,
    circle_thickness: int = 3
) -> np.ndarray:
    """
    Draw collision prediction visualization.
    
    Args:
        frame: Input frame
        current_point: Current (x, y) position of object
        predicted_points: List of predicted future (x, y) points
        time_to_collision: Time to collision in seconds
        color: RGB color for prediction visualization
        circle_radius: Radius of warning circle
        circle_thickness: Thickness of warning circle
        
    Returns:
        Frame with collision prediction visualization
    """
    # Draw warning circle around object
    cx, cy = current_point
    cv2.circle(frame, (cx, cy), circle_radius, color, circle_thickness)
    
    # Draw time to collision text
    ttc_text = f"TTC: {time_to_collision:.1f}s"
    draw_label(frame, cx - 40, cy - 30, ttc_text, color)
    
    # Draw predicted collision path
    for i in range(1, len(predicted_points)):
        pt1 = (int(predicted_points[i-1][0]), int(predicted_points[i-1][1]))
        pt2 = (int(predicted_points[i][0]), int(predicted_points[i][1]))
        cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
    
    return frame


def apply_walkable_area_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (230, 115, 0),  # BGR format
    alpha: float = 0.3
) -> np.ndarray:
    """
    Apply semi-transparent walkable area overlay.
    
    Args:
        frame: Input frame
        mask: Binary mask for walkable area
        color: BGR color for walkable area
        alpha: Transparency factor (0-1)
        
    Returns:
        Frame with walkable area overlay
    """
    # Create overlay
    overlay = frame.copy()
    overlay[mask > 0] = color
    
    # Blend with original frame
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return result


def draw_status_panel(
    frame: np.ndarray,
    action: str,
    warning: str,
    objects_count: int,
    collision_count: int = 0,
    show_tracking: bool = True,
    panel_height: int = 80
) -> np.ndarray:
    """
    Draw status panel at the bottom of the frame.
    
    Args:
        frame: Input frame
        action: Vehicle action text (e.g., "PROCEED", "STOP")
        warning: Warning message
        objects_count: Number of detected objects
        collision_count: Number of collision risks
        show_tracking: Whether tracking is enabled (affects display)
        panel_height: Height of the panel in pixels
        
    Returns:
        Frame with status panel
    """
    height, width = frame.shape[:2]
    
    # Create semi-transparent panel
    panel = frame.copy()
    cv2.rectangle(panel, (0, height - panel_height), (width, height), (0, 0, 0), -1)
    result = cv2.addWeighted(panel, 0.7, frame, 0.3, 0)
    
    # Define action color
    action_color = {
        "PROCEED": (0, 255, 0),  # Green
        "PROCEED WITH CAUTION": (0, 255, 255),  # Yellow
        "SLOW DOWN": (0, 165, 255),  # Orange
        "STOP": (0, 0, 255)  # Red
    }.get(action, (255, 255, 255))  # White as default
    
    # Draw action text
    cv2.putText(
        result,
        action,
        (20, height - panel_height + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        action_color,
        3
    )
    
    # Draw warning message
    cv2.putText(
        result,
        warning,
        (20, height - panel_height + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )
    
    # Draw object counts
    cv2.putText(
        result,
        f"Objects: {objects_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    if show_tracking:
        cv2.putText(
            result,
            f"Collision risks: {collision_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
    
    return result


def draw_legend(
    frame: np.ndarray,
    show_walkable_area: bool = True,
    show_collision_risks: bool = True
) -> np.ndarray:
    """
    Draw legend for visualization elements.
    
    Args:
        frame: Input frame
        show_walkable_area: Whether to include walkable area in legend
        show_collision_risks: Whether to include collision risks in legend
        
    Returns:
        Frame with legend
    """
    height, width = frame.shape[:2]
    
    # Draw legends at top-right
    if show_walkable_area:
        cv2.putText(
            frame,
            "Blue: Walkable Area",
            (width - 220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 115, 0),  # BGR format
            2
        )
    
    if show_collision_risks:
        cv2.putText(
            frame,
            "Red Circle: Collision Risk",
            (width - 270, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),  # BGR format
            2
        )
    
    return frame


def generate_random_color() -> Tuple[int, int, int]:
    """
    Generate a random RGB color.
    
    Returns:
        Random RGB color tuple
    """
    return (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255)
    )


def create_color_map(
    class_names: List[str],
    seed: int = 42
) -> Dict[str, Tuple[int, int, int]]:
    """
    Create a consistent color map for class names.
    
    Args:
        class_names: List of class names
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping class names to RGB colors
    """
    np.random.seed(seed)
    color_map = {}
    
    for name in class_names:
        color_map[name] = generate_random_color()
    
    return color_map


def visualize_system_results(
    frame: np.ndarray,
    objects: List[dict],
    walkable_area_mask: Optional[np.ndarray] = None,
    track_history: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    collision_objects: Optional[List[Tuple[dict, float]]] = None,
    action: str = "UNKNOWN",
    warning: str = "",
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    config: Optional[Any] = None
) -> np.ndarray:
    """
    Complete visualization of perception system results.
    
    Args:
        frame: Input frame
        objects: List of detected objects with bounding boxes, classes, etc.
        walkable_area_mask: Binary mask of walkable area
        track_history: Dictionary mapping track IDs to trajectory point histories
        collision_objects: List of collision objects with time to collision
        action: Current vehicle action recommendation
        warning: Warning message to display
        color_map: Dictionary mapping class names to colors
        config: Visualization configuration
        
    Returns:
        Frame with all visualizations applied
    """
    result_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Use default config if not provided
    if config is None:
        from .config import VisualizationConfig
        config = VisualizationConfig()
    
    # Generate color map if not provided
    if color_map is None:
        unique_classes = set(obj['class_name'] for obj in objects)
        color_map = create_color_map(list(unique_classes))
    
    # Apply walkable area overlay
    if walkable_area_mask is not None and config.show_road:
        result_frame = apply_walkable_area_overlay(result_frame, walkable_area_mask)
    
    # Process collision objects into a lookup dict for quick access
    collision_lookup = {}
    if collision_objects:
        collision_lookup = {obj[0]['track_id']: obj[1] for obj in collision_objects if 'track_id' in obj[0]}
    
    # Draw object detections and trajectories
    for obj in objects:
        # Determine if object is tracked
        is_tracked = 'track_id' in obj and obj['track_id'] is not None
        
        # Get object info
        class_name = obj['class_name']
        confidence = obj['confidence']
        x1, y1, x2, y2 = obj['bbox']
        
        # Get color for this object (from track_id or class_name)
        if is_tracked:
            track_id = obj['track_id']
            # Generate color based on track_id if not in color map
            if track_id not in color_map:
                color_map[track_id] = generate_random_color()
            color = color_map[track_id]
        else:
            # Use class color
            color = color_map.get(class_name, (0, 255, 0))
        
        # Draw bounding box
        draw_bounding_box(result_frame, x1, y1, x2, y2, color)
        
        # Calculate object center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Draw label
        if config.show_labels:
            if is_tracked:
                label = f"{class_name} #{track_id} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"
            
            draw_label(result_frame, x1, y1, label, (255, 255, 255), color)
        
        # Draw trajectory if available and tracking is enabled
        if is_tracked and track_history and track_id in track_history and config.show_tracks:
            draw_trajectory(result_frame, list(track_history[track_id]), color)
            
            # Check if this object is on collision course
            if track_id in collision_lookup:
                ttc = collision_lookup[track_id]
                
                # Calculate future trajectory points based on recent movement
                points = list(track_history[track_id])
                if len(points) >= 5:
                    # Calculate average movement vector
                    dx = sum(points[i][0] - points[i-1][0] for i in range(-1, -5, -1)) / 4
                    dy = sum(points[i][1] - points[i-1][1] for i in range(-1, -5, -1)) / 4
                    
                    # Project forward
                    future_points = [(cx + dx*t, cy + dy*t) for t in range(1, 21)]
                    
                    # Draw collision prediction
                    if config.show_collision_prediction:
                        draw_collision_prediction(
                            result_frame, 
                            (cx, cy), 
                            future_points, 
                            ttc
                        )
    
    # Draw status panel
    result_frame = draw_status_panel(
        result_frame,
        action,
        warning,
        len(objects),
        len(collision_objects) if collision_objects else 0,
        config.show_tracks,
        config.panel_height
    )
    
    # Draw legend
    result_frame = draw_legend(
        result_frame,
        config.show_road,
        config.show_collision_prediction and collision_objects
    )
    
    return result_frame
