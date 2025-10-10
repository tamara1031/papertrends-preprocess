"""
Score calculation utilities for clustering evaluation.
This module provides functions to compute clustering quality scores with minimal memory usage.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from hdbscan import validity_index
from bertopic import BERTopic

try:
    from .memory_utils import force_memory_cleanup
except ImportError:
    from memory_utils import force_memory_cleanup


def compute_silhouette_score(
    labels: np.ndarray,
    embeddings: np.ndarray
) -> float:
    """Compute silhouette score using embeddings with minimal memory usage."""
    try:
        # Early validation to avoid unnecessary processing
        if embeddings is None:
            return 0.0
        
        # Filter out noise points (-1 labels) - create mask only
        valid_mask = labels != -1
        valid_count = np.sum(valid_mask)
        
        if valid_count < 2:
            return 0.0  # Not enough valid points
            
        # Extract only valid data (avoid copying large arrays)
        valid_labels = labels[valid_mask]
        valid_embeddings = embeddings[valid_mask]
        
        # Check if we have multiple clusters
        unique_labels = np.unique(valid_labels)
        if len(unique_labels) < 2:
            return 0.0  # Need at least 2 clusters for silhouette score
        
        # Compute silhouette score using euclidean metric on embeddings
        silhouette_avg = silhouette_score(
            valid_embeddings, 
            valid_labels, 
            metric='euclidean'
        )
        
        # Normalize to [0, 1] range
        silhouette_score_normalized = (silhouette_avg + 1) / 2
        
        # Clean up intermediate variables
        del valid_labels, valid_embeddings, valid_mask
        
        return silhouette_score_normalized
        
    except KeyboardInterrupt:
        raise    
    except Exception as e:
        print(f"Warning: Silhouette score computation failed: {e}")
        return 0.0  # Return neutral score on error


def compute_dbcv_score(
    labels: np.ndarray,
    embeddings: np.ndarray
) -> float:
    """Compute DBCV score using PCA with minimal memory usage."""
    try:
        # Early validation to avoid unnecessary processing
        valid_mask = labels != -1
        valid_count = np.sum(valid_mask)
        
        if valid_count < 2:
            return 0.0  # Not enough valid points
            
        # Extract only valid labels (small array)
        valid_labels = labels[valid_mask]
        
        # Check if we have multiple clusters before processing embeddings
        unique_labels = np.unique(valid_labels)
        if len(unique_labels) < 2:
            return 0.0  # Need at least 2 clusters for DBCV
 
        valid_embeddings = embeddings[valid_mask]
        
        # Memory-efficient PCA: Use fewer components for large datasets

        pca = PCA(n_components=0.99, random_state=42)
        
        # Transform embeddings
        projected_embeddings = pca.fit_transform(valid_embeddings)
        
        # Ensure float64 for HDBSCAN compatibility
        if projected_embeddings.dtype != np.float64:
            projected_embeddings = projected_embeddings.astype(np.float64)
        
        # Compute DBCV using cosine metric
        dbcv_score = validity_index(projected_embeddings, valid_labels, metric='cosine')
        
        # Normalize to [0, 1] range
        dbcv_score_normalized = (dbcv_score + 1) / 2
        
        # Clean up all intermediate variables immediately
        del projected_embeddings, valid_embeddings, valid_labels, valid_mask, pca
        
        return dbcv_score_normalized
        
    except KeyboardInterrupt:
        raise    
    except Exception as e:
        print(f"Warning: DBCV score computation failed: {e}")
        return 0.0  # Return neutral score on error


if __name__ == "__main__":
    # Test score utilities
    print("Score utilities test:")
    
    # Create test data
    labels = np.array([0, 0, 1, 1, 2, 2, -1, -1])
    umap_embedding = np.random.rand(8, 2)
    original_embeddings = np.random.rand(8, 768)
    
    # Test silhouette score
    silhouette_score = compute_silhouette_score(labels, umap_embedding)
    print(f"Silhouette score: {silhouette_score:.4f}")
    
    # Test DBCV score
    dbcv_score = compute_dbcv_score(labels, original_embeddings)
    print(f"DBCV score: {dbcv_score:.4f}")
    
    print("Score utilities test completed successfully!")
