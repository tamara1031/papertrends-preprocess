"""
Simplified utilities for papertrends preprocessing.
Consolidated essential functions with minimal complexity.
"""

import gc
import numpy as np
from sklearn.metrics import silhouette_score
from hdbscan import validity_index
import torch

def compute_silhouette_score(labels: np.ndarray, embeddings: np.ndarray) -> float:
    """Compute silhouette score with error handling."""
    try:
        # Filter out noise points
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            return -1.0
            
        valid_labels = labels[valid_mask]
        valid_embeddings = embeddings[valid_mask]
        
        # Check for multiple clusters
        if len(np.unique(valid_labels)) < 2:
            return -1.0
            
        return silhouette_score(valid_embeddings, valid_labels, metric='euclidean')
    except Exception:
        return -1.0

def compute_dbcv_score(labels: np.ndarray, embeddings: np.ndarray) -> float:
    """Compute DBCV score with error handling."""
    try:
        # Filter out noise points
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            return -1.0
            
        valid_labels = labels[valid_mask]
        valid_embeddings = embeddings[valid_mask].astype(np.float64)
        
        # Check for multiple clusters
        if len(np.unique(valid_labels)) < 2:
            return -1.0
            
        return validity_index(valid_embeddings, valid_labels, metric='euclidean')
    except Exception:
        return -1.0

def cleanup_memory():
    """Simple memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_info() -> dict:
    """Get basic memory information."""
    info = {}
    if torch.cuda.is_available():
        info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024
        info['gpu_cached'] = torch.cuda.memory_reserved() / 1024 / 1024
    return info
