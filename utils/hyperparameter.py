from typing import List, Union, Dict, Any
from dataclasses import dataclass

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Hyperparameters:
    """Complete hyperparameter set for BERTopic clustering."""
    
    # Topic representation
    top_n_words: int
    
    # Text vectorization
    ngram_range: List[int]
    min_df: Union[float, int]
    max_df: Union[float, int]
    
    # UMAP dimensionality reduction
    n_neighbors: int
    n_components: int
    umap_metric: str
    
    # HDBSCAN clustering
    min_cluster_size: int
    min_samples: int
    hdbscan_metric: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'top_n_words': self.top_n_words,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'n_neighbors': self.n_neighbors,
            'n_components': self.n_components,
            'umap_metric': self.umap_metric,
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'hdbscan_metric': self.hdbscan_metric
        }
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'Hyperparameters':
        """Create Hyperparameters instance from dictionary (e.g., Optuna best_params)."""
        # Calculate min_samples from min_samples_multiplier and min_cluster_size
        min_cluster_size = params_dict['min_cluster_size']
        min_samples_multiplier = params_dict['min_samples_multiplier']
        min_samples = max(1, min(int(min_cluster_size * min_samples_multiplier), min_cluster_size))
        
        return cls(
            top_n_words=params_dict['top_n_words'],
            ngram_range=params_dict['ngram_range'],
            min_df=params_dict['min_df'],
            max_df=params_dict['max_df'],
            n_neighbors=params_dict['n_neighbors'],
            n_components=params_dict['n_components'],
            umap_metric=params_dict['umap_metric'],
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            hdbscan_metric=params_dict['hdbscan_metric']
        )