import yaml
from typing import List, Dict
from pathlib import Path


def load_categories_from_yaml(yaml_path: str = "config/categories.yaml") -> Dict[str, str]:
    """
    Load categories from YAML file and return as dictionary.
    
    Args:
        yaml_path: Path to the categories YAML file
        
    Returns:
        Dictionary mapping category codes to descriptions
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Categories file not found: {yaml_path}")
    
    with open(yaml_file, 'r', encoding='utf-8') as file:
        categories = yaml.safe_load(file)
    
    return categories


def get_category_codes(yaml_path: str = "config/categories.yaml") -> List[str]:
    """
    Get list of category codes from YAML file.
    
    Args:
        yaml_path: Path to the categories YAML file
        
    Returns:
        List of category codes
    """
    categories = load_categories_from_yaml(yaml_path)
    return list(categories.keys())


def get_category_descriptions(yaml_path: str = "config/categories.yaml") -> List[str]:
    """
    Get list of category descriptions from YAML file.
    
    Args:
        yaml_path: Path to the categories YAML file
        
    Returns:
        List of category descriptions
    """
    categories = load_categories_from_yaml(yaml_path)
    return list(categories.values())


def get_categories_by_prefix(prefix: str, yaml_path: str = "config/categories.yaml") -> Dict[str, str]:
    """
    Get categories filtered by prefix (e.g., 'cs.' for computer science categories).
    
    Args:
        prefix: Category prefix to filter by
        yaml_path: Path to the categories YAML file
        
    Returns:
        Dictionary of filtered categories
    """
    categories = load_categories_from_yaml(yaml_path)
    return {k: v for k, v in categories.items() if k.startswith(prefix)}
