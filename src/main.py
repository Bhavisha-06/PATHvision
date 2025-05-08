#!/usr/bin/env python3
"""
Main entry point for the Advanced Vehicle Perception System
"""

import argparse
import os
import sys
import yaml

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from perception.system import VehiclePerceptionSystem
from utils.config import load_config, merge_configs


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Vehicle Perception System")
    parser.add_argument("--input", type=str, default=None, 
                        help="Input video file path")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output video file path")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to configuration file (YAML)")
    parser.add_argument("--no-tracking", action="store_true", 
                        help="Disable object tracking")
    parser.add_argument("--frame-skip", type=int, default=None, 
                        help="Process road detection every N frames")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to YOLO model")
    parser.add_argument("--conf-thresh", type=float, default=None, 
                        help="Detection confidence threshold")
    parser.add_argument("--use-detectron", action="store_true", 
                        help="Force use of Detectron2 for road segmentation")
    parser.add_argument("--device", type=str, default=None, 
                        help="Processing device ('cpu', 'cuda', '0', '1', etc.)")
    
    return parser.parse_args()


def prepare_config(args):
    """Prepare configuration by loading default config and overriding with args."""
    # Load default configuration
    config_path = os.path.join(os.path.dirname(__file__), "config", "default_config.yaml")
    config = load_config(config_path)
    
    # If a custom config is provided, load and merge it
    if args.config:
        custom_config = load_config(args.config)
        config = merge_configs(config, custom_config)
    
    # Override config with command-line arguments
    if args.no_tracking is not None:
        config["tracking"]["enabled"] = not args.no_tracking
    if args.frame_skip is not None:
        config["road_detection"]["frame_skip"] = args.frame_skip
    if args.model is not None:
        config["detection"]["model"] = args.model
    if args.conf_thresh is not None:
        config["detection"]["confidence_threshold"] = args.conf_thresh
    if args.use_detectron:
        config["road_detection"]["use_detectron"] = "force"
    if args.device is not None:
        config["system"]["device"] = args.device
    
    return config


def main():
    """Main function."""
    args = parse_arguments()
    config = prepare_config(args)
    
    # Initialize the system
    system = VehiclePerceptionSystem(config)
    
    # Determine input and output paths
    input_path = args.input
    if not input_path:
        print("Error: Input video path is required.")
        return 1
    
    output_path = args.output
    if not output_path:
        # Create default output path based on input path
        basename = os.path.basename(input_path)
        filename, ext = os.path.splitext(basename)
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{filename}_processed{ext}")
    
    # Process the video
    system.process_video(
        input_path=input_path,
        output_path=output_path
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
