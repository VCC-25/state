"""
Device configuration helper for STATE embedding training.
Automatically detects and configures settings for MPS (Apple Silicon), GPU, or CPU.
"""

import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def detect_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_config() -> Dict[str, Any]:
    """Get device-specific configuration."""
    device = detect_device()
    
    config = {
        "device": device,
        "devices": "auto",
        "strategy": "auto",
        "accelerator": "auto",
        "precision": "auto",
        "mixed_precision": True,
        "compile_model": False,
    }
    
    if device == "cuda":
        # GPU configuration
        config.update({
            "devices": 1,  # Use single GPU
            "strategy": "auto",
            "accelerator": "gpu",
            "precision": "16-mixed",  # Mixed precision for GPU
            "mixed_precision": True,
            "compile_model": True,  # Enable torch.compile for GPU
        })
        logger.info("✅ Using CUDA GPU configuration")
        
    elif device == "mps":
        # Apple Silicon configuration
        config.update({
            "devices": 1,  # Use single MPS device
            "strategy": "auto",
            "accelerator": "mps",
            "precision": "32-true",  # Full precision for MPS stability
            "mixed_precision": False,  # Disable mixed precision for MPS
            "compile_model": False,  # Disable torch.compile for MPS compatibility
        })
        logger.info("✅ Using MPS (Apple Silicon) configuration")
        
    else:
        # CPU configuration
        config.update({
            "devices": 1,
            "strategy": "auto",
            "accelerator": "cpu",
            "precision": "32-true",
            "mixed_precision": False,
            "compile_model": False,
        })
        logger.info("✅ Using CPU configuration")
    
    return config

def update_config_for_device(config: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with device-specific settings."""
    device_config = get_device_config()
    
    # Update training section
    if "training" not in config:
        config["training"] = {}
    
    config["training"].update({
        "devices": device_config["devices"],
        "strategy": device_config["strategy"],
        "accelerator": device_config["accelerator"],
        "precision": device_config["precision"],
    })
    
    # Update experiment section
    if "experiment" not in config:
        config["experiment"] = {}
    
    config["experiment"].update({
        "device_type": device_config["device"],
        "mixed_precision": device_config["mixed_precision"],
        "compile_model": device_config["compile_model"],
    })
    
    # Update model section if it exists
    if "model" in config:
        config["model"]["compile"] = device_config["compile_model"]
    
    logger.info(f"✅ Updated configuration for {device_config['device']} device")
    return config

def print_device_info():
    """Print detailed device information."""
    logger.info("🔍 Device Detection Results:")
    logger.info(f"  - CUDA available: {torch.cuda.is_available()}")
    logger.info(f"  - MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"  - Selected device: {detect_device()}")
    
    if torch.cuda.is_available():
        logger.info(f"  - CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"  - CUDA device name: {torch.cuda.get_device_name(0)}")
    
    if torch.backends.mps.is_available():
        logger.info("  - MPS (Apple Silicon) detected")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_device_info()
    config = get_device_config()
    logger.info(f"Device configuration: {config}") 