"""
Score calculation utilities for clustering evaluation.
This module provides functions to compute clustering quality scores with minimal memory usage.
"""

import numpy as np
from sklearn.decomposition import PCA
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
        return 0.0  # Not enough valid points
        
    # Extract only valid data
    valid_labels = labels[valid_mask]
    valid_embeddings = embeddings[valid_mask]
    
    # Check if we have multiple clusters
    unique_labels = np.unique(valid_labels)
    if len(unique_labels) < 2:
        return 0.0  # Need at least 2 clusters for silhouette score
    
    # Compute silhouette score using euclidean metric
    silhouette_avg = silhouette_score(
        valid_embeddings, 
        valid_labels, 
        metric='euclidean'
    )
    
    return silhouette_avg


def compute_dbcv_score(
    labels: np.ndarray,
    embeddings: np.ndarray
) -> float:
    """Compute DBCV score using PCA with minimal memory usage."""
    # Filter out noise points (-1 labels)
    valid_mask = labels != -1
    valid_count = np.sum(valid_mask)
    
    if valid_count < 2:
        return 0.0  # Not enough valid points
        
    # Extract only valid labels
    valid_labels = labels[valid_mask]
    
    # Check if we have multiple clusters
    unique_labels = np.unique(valid_labels)
    if len(unique_labels) < 2:
        return 0.0  # Need at least 2 clusters for DBCV

    valid_embeddings = embeddings[valid_mask]
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.99, random_state=42)
    projected_embeddings = pca.fit_transform(valid_embeddings)
    
    # Ensure float64 for HDBSCAN compatibility
    if projected_embeddings.dtype != np.float64:
        projected_embeddings = projected_embeddings.astype(np.float64)
    
    # Compute DBCV using cosine metric
    dbcv_score = validity_index(projected_embeddings, valid_labels, metric='cosine')
    
    return dbcv_score


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
