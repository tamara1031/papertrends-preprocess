import yaml
from typing import List, Dict
from pathlib import Path


def load_categories_from_yaml(yaml_path: str = "config/categories.yaml") -> Dict[str, str]:
    """
    Load categories from YAML file and return as flattened dictionary.
    
    Args:
        yaml_path: Path to the categories YAML file
        
    Returns:
        Dictionary mapping category codes to descriptions
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Categories file not found: {yaml_path}")
    
    with open(yaml_file, 'r', encoding='utf-8') as file:
        nested_categories = yaml.safe_load(file)
    
    # Flatten the nested structure
    categories = {}
    if isinstance(nested_categories, dict):
        for main_category, subcategories in nested_categories.items():
            if isinstance(subcategories, dict):
                categories.update(subcategories)
    
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


def get_categories_by_main_category(main_category: str, yaml_path: str = "config/categories.yaml") -> Dict[str, str]:
    """
    Get all subcategories under a main category (e.g., 'cs', 'physics', 'math').
    
    Args:
        main_category: Main category name (e.g., 'cs', 'physics', 'math')
        yaml_path: Path to the categories YAML file
        
    Returns:
        Dictionary of subcategories under the main category
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Categories file not found: {yaml_path}")
    
    with open(yaml_file, 'r', encoding='utf-8') as file:
        nested_categories = yaml.safe_load(file)
    
    if not isinstance(nested_categories, dict):
        return {}
    
    return nested_categories.get(main_category, {})


def get_main_categories(yaml_path: str = "config/categories.yaml") -> List[str]:
    """
    Get list of main categories from YAML file.
    
    Args:
        yaml_path: Path to the categories YAML file
        
    Returns:
        List of main category names
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Categories file not found: {yaml_path}")
    
    with open(yaml_file, 'r', encoding='utf-8') as file:
        nested_categories = yaml.safe_load(file)
    
    if not isinstance(nested_categories, dict):
        return []
    
    return list(nested_categories.keys())
