"""
Configuration loader for Banner Brand Tagger.
Loads configuration from JSON file with environment variable overrides.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_file: str = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to config file. If None, uses default location.
    
    Returns:
        Dictionary containing configuration values.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file is invalid JSON.
    """
    if config_file is None:
        # Default to config file in src/config directory
        current_file = Path(__file__)
        config_dir = current_file.parent.parent / "config"
        config_file = config_dir / "banner_brand_tagger_config.json"
    else:
        config_file = Path(config_file)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        
        logger.info(f"✅ Loaded configuration from: {config_file}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in config file: {e}", exc_info=True)
        raise ValueError(f"Invalid JSON in config file: {e}") from e
    except Exception as e:
        logger.error(f"❌ Failed to load config file: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load config file: {e}") from e


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None, env_var: str = None) -> Any:
    """
    Get a configuration value from nested dictionary using dot notation.
    Optionally override with environment variable.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the value (e.g., "model.name").
        default: Default value if key not found.
        env_var: Environment variable name to check first (optional).
    
    Returns:
        Configuration value or default.
    """
    # Check environment variable first if provided
    if env_var:
        env_value = os.getenv(env_var)
        if env_value is not None:
            logger.debug(f"Using environment variable {env_var} instead of config value")
            return env_value
    
    # Navigate through nested dictionary
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if default is not None:
                logger.debug(f"Config key '{key_path}' not found, using default value")
                return default
            raise KeyError(f"Config key '{key_path}' not found")
    
    return value
