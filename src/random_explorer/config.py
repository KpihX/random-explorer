import yaml

from pathlib import Path
from functools import lru_cache

from .utils import Console

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
console = Console()

@lru_cache(maxsize=1)
def load_config(config_path=CONFIG_PATH, **overrides):
    """Load local config with eventual overrides"""
    if not config_path.exists():
        console.display_error(f"File not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError:
            console.print_exception()
            return None
        
    config.update(overrides)
    return config
