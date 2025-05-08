import numpy as np

# Constants for collision detection
TIME_TO_COLLISION_THRESHOLD = 3.0  # Time to collision threshold in seconds
DANGER_THRESHOLD = 0.6  # Threshold for considering an object dangerous

# Classes that can cause obstacles
OBSTACLE_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
                   'dog', 'cow', 'elephant', 'bear', 'horse', 'sheep']

class CollisionPredictor:
    def __init__(self, obstacle_classes=None):
        """
        Initialize the collision predictor.
        
        Args:
            obstacle_classes: List of classes to consider as obstacles
        """
        self.obstacle_classes = obstacle_classes or OBSTACLE_CLASSES
    
    def is_on_collision_course(self, track, frame_shape):
        """
        Determine if an object is on a collision course with the vehicle.
        
        Args:
            track: List of (x, y) coordinates representing the object's path
            frame_shape: Shape of the video frame (height, width, channels)
            
        Returns:
            (danger_value, time_to_collision): Tuple of danger value (0-1) 
            and estimated time to collision in seconds
        """
        if len(track) < 5:  # Need at least 5 points for reliable prediction
            return False, float('inf')
        
        # Get current position (last point in the track)
        current_x, current_y = track[-1]
        
        # Calculate average movement vector from the last 5 positions
        dx = sum(track[i][0] - track[i - 1][0] for i in range(-1, -5, -1)) / 4
        dy = sum(track[i][1] - track[i - 1][1] for i in range(-1, -5, -1)) / 4
        
        # Project the path forward
        height, width = frame_shape[:2]
        vehicle_center_x = width // 2
        vehicle_position_y = height  # Bottom of the frame
        
        # If the object is not moving or moving away (dy <= 0), it's not on collision course
        if dy <= 0:
            return False, float('inf')
        
        # Calculate time steps needed for the object to reach the vehicle's y position
        steps_to_collision = (vehicle_position_y - current_y) / dy if dy != 0 else float('inf')
        
        # Calculate the x position at collision
        collision_x = current_x + dx * steps_to_collision
        
        # Check if the collision x position is within the vehicle's width (middle third of the frame)
        vehicle_width_start = width // 3
        vehicle_width_end = 2 * width // 3
        
        # Calculate distance to vehicle center at collision point
        distance_to_center = abs(collision_x - vehicle_center_x)
        normalized_distance = distance_to_center / (width / 2)  # Normalize to [0, 1]
        
        # Calculate time to collision in seconds (assuming frame rate)
        fps = 30  # Estimate, will be replaced with actual fps
        time_to_collision = steps_to_collision / fps
        
        # Object is on collision course if it will intersect with vehicle area
        is_colliding = vehicle_width_start <= collision_x <= vehicle_width_end
        
        # If it's close to the vehicle area, consider it a potential collision
        danger_value = 1.0 - normalized_distance if normalized_distance < 0.5 else 0.0
        
        return danger_value > DANGER_THRESHOLD, time_to_collision
    
    def determine_action(self, objects, track_history, walkable_area_mask, frame_shape, fps=30):
        """
        Determine what action the vehicle should take based on detected objects, 
        their trajectories, and walkable area.
        
        Args:
            objects: List of detected/tracked objects
            track_history: Dictionary of track histories
            walkable_area_mask: Binary mask of walkable area
            frame_shape: Shape of the video frame
            fps: Frames per second of the video
            
        Returns:
            (action, confidence, warning, collision_objects): Tuple with:
            - action: String with recommended action
            - confidence: Confidence level (0-1)
            - warning: String with warning message
            - collision_objects: List of objects on collision course
        """
        height, width = frame_shape[:2]
        
        # Check if there's any walkable area
        walkable_percentage = np.sum(walkable_area_mask > 0) / (height * width) * 100
        
        # Find objects on collision course (only for tracked objects)
        collision_objects = []
        min_time_to_collision = float('inf')
        critical_object = None
        
        tracked_objects = [obj for obj in objects if obj[0] == "tracked"]
        
        for obj in tracked_objects:
            _, class_name, track_id, x1, y1, x2, y2, conf = obj
            
            # Skip if not an obstacle class
            if class_name not in self.obstacle_classes:
                continue
            
            # Check if we have track history
            if track_id in track_history and len(track_history[track_id]) >= 5:
                # Check if object is on collision course
                collision_danger, time_to_collision = self.is_on_collision_course(
                    list(track_history[track_id]), frame_shape
                )
                
                if collision_danger:
                    collision_objects.append((obj, time_to_collision))
                    
                    if time_to_collision < min_time_to_collision:
                        min_time_to_collision = time_to_collision
                        critical_object = obj
        
        # Determine action based on collision risks and walkable area
        if not walkable_percentage or walkable_percentage < 5:
            return "STOP", 0.95, "No walkable area detected", collision_objects
        
        if collision_objects:
            # Sort collisions by time (most imminent first)
            collision_objects.sort(key=lambda x: x[1])
            
            # Get the most imminent collision
            critical_obj, ttc = collision_objects[0]
            class_name = critical_obj[1]  # class_name is at index 1 for tracked objects
            
            if ttc < 1.0:  # Very imminent collision (less than 1 second)
                return "STOP", 0.99, f"IMMINENT COLLISION with {class_name} in {ttc:.1f}s", collision_objects
            elif ttc < 2.0:  # Collision within 2 seconds
                return "STOP", 0.95, f"Critical {class_name} approaching in {ttc:.1f}s", collision_objects
            elif ttc < 3.0:  # Collision within 3 seconds
                return "SLOW DOWN", 0.90, f"{class_name} on collision course in {ttc:.1f}s", collision_objects
            else:  # Collision more than 3 seconds away
                return "PROCEED WITH CAUTION", 0.80, f"{class_name} approaching in {ttc:.1f}s", collision_objects
        
        # Check for static obstacles in the immediate path
        obstacles_in_path = False
        obstacle_distance = float('inf')
        static_critical_object = None
        
        # Calculate the center path of the vehicle (middle third of bottom part of image)
        center_left = width // 3
        center_right = 2 * width // 3
        
        for obj in objects:
            if obj[0] == "tracked":
                _, class_name, track_id, x1, y1, x2, y2, conf = obj
            else:
                _, class_name, _, x1, y1, x2, y2, conf = obj
            
            # Skip if not an obstacle class
            if class_name not in self.obstacle_classes:
                continue
            
            # Object center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Check if object is in the path and how close it is
            if center_left <= cx <= center_right:
                # Calculate distance from bottom (closer to vehicle)
                distance = height - cy
                
                if distance < obstacle_distance:
                    obstacle_distance = distance
                    obstacles_in_path = True
                    static_critical_object = obj
        
        # Determine action based on static obstacles and walkable area
        if obstacles_in_path:
            class_name = static_critical_object[1]  # class_name is at index 1
            distance_percentage = obstacle_distance / height * 100
            
            if distance_percentage < 20:  # Very close
                return "STOP", 0.95, f"Immediate {class_name} in path", []
            elif distance_percentage < 40:  # Moderately close
                return "SLOW DOWN", 0.85, f"{class_name} ahead", []
            else:  # Further away
                return "PROCEED WITH CAUTION", 0.75, f"{class_name} detected", []
        
        # No obstacles in direct path
        if walkable_percentage > 30:
            return "PROCEED", 0.90, "Clear path ahead", []
        elif walkable_percentage > 15:
            return "PROCEED WITH CAUTION", 0.80, "Limited walkable area", []
        else:
            return "SLOW DOWN", 0.85, "Restricted walkable area", []
