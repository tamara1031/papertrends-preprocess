"""
Score calculation utilities for clustering evaluation.
This module provides functions to compute clustering quality scores with minimal memory usage.
"""

import numpy as np
from sklearn.metrics import silhouette_score
from hdbscan import validity_index


def compute_silhouette_score(
    labels: np.ndarray,
    embeddings: np.ndarray
) -> float:
    """Compute silhouette score using embeddings with minimal memory usage."""
    # Filter out noise points (-1 labels)
    valid_mask = labels != -1
    valid_count = np.sum(valid_mask)
    
    if valid_count < 2:
        raise ValueError("Not enough valid points for silhouette score calculation")
        
    # Extract only valid data
    valid_labels = labels[valid_mask]
    valid_embeddings = embeddings[valid_mask]
    
    # Check if we have multiple clusters
    unique_labels = np.unique(valid_labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 clusters for silhouette score calculation")
    
    # Compute silhouette score using euclidean metric
    silhouette_result = silhouette_score(
        valid_embeddings, 
        valid_labels, 
        metric='euclidean'
    )
    
    return silhouette_result


def compute_dbcv_score(
    labels: np.ndarray,
    embeddings: np.ndarray
) -> float:
    """Compute DBCV score."""
    
    return validity_index(embeddings.astype(np.float64), labels,  metric='euclidean')


if __name__ == "__main__":
    # Test score utilities
    print("Score utilities test:")
    
    # Create test data
    labels = np.array([0, 0, 1, 1, 2, 2, -1, -1])
    umap_embedding = np.random.rand(8, 2)
    original_embeddings = np.random.rand(8, 768)
    
    # Test silhouette score
    try:
        silhouette_result = compute_silhouette_score(labels, umap_embedding)
        print(f"Silhouette score: {silhouette_result:.4f}")
    except ValueError as e:
        print(f"Silhouette score error: {e}")
    
    # Test DBCV score
    try:
        dbcv_score = compute_dbcv_score(labels, original_embeddings)
        print(f"DBCV score: {dbcv_score:.4f}")
    except ValueError as e:
        print(f"DBCV score error: {e}")
    
    print("Score utilities test completed successfully!")
