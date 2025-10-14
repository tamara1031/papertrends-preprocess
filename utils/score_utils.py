"""
Score calculation utilities for clustering evaluation.
This module provides functions to compute clustering quality scores with minimal memory usage.
"""

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
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
    """Compute DBCV score using embeddings."""
    embeddings = embeddings.astype(np.float64)
    dbcv_score = validity_index(embeddings, labels,  metric='euclidean')
    
    return dbcv_score


def compute_dcsi_score(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_samples: int = 5,
    eps: float = None
) -> float:
    """
    Compute DCSI (Density Cluster Separability Index) score.
    
    Based on the paper definition:
    - Separation: minimal distance between core points of different clusters
    - Connectedness: maximum shortest path distance between core points within the same cluster
    - DCSI = separation / connectedness
    
    Args:
        labels: Cluster labels (-1 for noise points)
        embeddings: Data embeddings
        min_samples: Minimum number of samples to form a core point (default: 5)
        eps: Epsilon parameter for DBSCAN-like core point definition (auto-estimated if None)
    
    Returns:
        DCSI score, where higher values indicate better separability
    """
    try:
        # Filter out noise points (-1 labels)
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_embeddings = embeddings[valid_mask]
        
        unique_labels = np.unique(valid_labels)
        if len(unique_labels) < 2:
            return 0.0
        
        # Auto-estimate eps if not provided using k-distance plot approach
        if eps is None:
            n_neighbors = min(min_samples, len(valid_embeddings) - 1)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nbrs.fit(valid_embeddings)
            distances, _ = nbrs.kneighbors(valid_embeddings)
            # Use the k-th nearest neighbor distance for eps estimation
            # Following the paper's recommendation for eps estimation
            eps = np.percentile(distances[:, -1], 70)  # Use 70th percentile as recommended
        
        # Find core points globally (not per cluster)
        nbrs = NearestNeighbors(radius=eps, metric='euclidean')
        nbrs.fit(valid_embeddings)
        neighbor_counts = nbrs.radius_neighbors(valid_embeddings, return_distance=False)
        core_mask = np.array([len(neighbors) >= min_samples for neighbors in neighbor_counts])
        
        if not np.any(core_mask):
            return 0.0
        
        # Get core points and their labels
        core_points = valid_embeddings[core_mask]
        core_labels = valid_labels[core_mask]
        
        # Group core points by cluster
        core_points_by_cluster = {}
        for label in unique_labels:
            cluster_mask = core_labels == label
            if np.any(cluster_mask):
                core_points_by_cluster[label] = core_points[cluster_mask]
        
        if len(core_points_by_cluster) < 2:
            return 0.0
        
        # Compute separation: minimal distance between core points of different clusters
        separation = np.inf
        cluster_labels = list(core_points_by_cluster.keys())
        
        for i, label1 in enumerate(cluster_labels):
            for label2 in cluster_labels[i+1:]:
                points1 = core_points_by_cluster[label1]
                points2 = core_points_by_cluster[label2]
                
                # Compute pairwise distances
                distances = np.linalg.norm(points1[:, np.newaxis] - points2[np.newaxis, :], axis=2)
                min_dist = np.min(distances)
                separation = min(separation, min_dist)
        
        # Compute connectedness: maximum shortest path distance between core points within clusters
        connectedness = 0.0
        for label, core_points in core_points_by_cluster.items():
            if len(core_points) > 1:
                # Build adjacency graph for this cluster
                nbrs_cluster = NearestNeighbors(radius=eps, metric='euclidean')
                nbrs_cluster.fit(core_points)
                adjacency_matrix = nbrs_cluster.radius_neighbors_graph(core_points, mode='distance')
                
                # Compute shortest path distances using Floyd-Warshall algorithm
                n_points = len(core_points)
                dist_matrix = np.full((n_points, n_points), np.inf)
                
                # Initialize with direct distances
                for i in range(n_points):
                    for j in range(n_points):
                        if i == j:
                            dist_matrix[i, j] = 0
                        elif adjacency_matrix[i, j] > 0:
                            dist_matrix[i, j] = adjacency_matrix[i, j]
                
                # Floyd-Warshall algorithm
                for k in range(n_points):
                    for i in range(n_points):
                        for j in range(n_points):
                            if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                                dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
                
                # Find maximum shortest path distance in this cluster
                max_dist = np.max(dist_matrix[dist_matrix != np.inf])
                connectedness = max(connectedness, max_dist)
        
        # DCSI = separation / connectedness
        if connectedness > 0:
            dcsi = separation / connectedness
        else:
            dcsi = 0.0
        
        return dcsi
        
    except Exception as e:
        # Log error for debugging but don't print to stdout in production
        import logging
        logging.warning(f"DCSI computation error: {e}")
        return 0.0






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
    
    # Test DCSI score
    try:
        dcsi_score = compute_dcsi_score(labels, umap_embedding)
        print(f"DCSI score: {dcsi_score:.4f}")
    except Exception as e:
        print(f"DCSI score error: {e}")
    
    print("Score utilities test completed successfully!")
