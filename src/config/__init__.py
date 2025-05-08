"""
Configuration files for the Vehicle Perception System.

This package contains configuration files and loaders for the system.
"""

import os
import yaml

def get_default_config_path():
    """Returns the path to the default configuration file."""
    return os.path.join(os.path.dirname(__file__), 'default_config.yaml')

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str, optional): Path to config file. If None, uses default config.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = get_default_config_path()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

__all__ = ['get_default_config_path', 'load_config']
