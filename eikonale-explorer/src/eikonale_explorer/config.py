import yaml
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict

CONFIG_FILE = Path(__file__).parent / "config.yaml"

@lru_cache()
def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        return {}
        
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value by key."""
    config = load_config()
    return config.get(key, default)
