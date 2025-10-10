from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import gc
import os
import pickle
import json
import numpy as np
import warnings
import psutil
import sys

import torch
import gc

import optuna
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import optuna.exceptions
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN, validity_index

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model, CustomEmbeddingModel, get_category_codes
from memory_utils import (
    log_memory_usage, force_memory_cleanup, check_memory_threshold,
    get_dataset_memory_estimate, recommend_dataset_limit
)

# Suppress expected numerical warnings (validated as safe)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='hdbscan.validity')
warnings.filterwarnings('ignore', message='overflow encountered in power')
warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

# ============================================================================
# Memory Management
# ============================================================================

# Memory management functions are now imported from memory_utils

def cleanup_bertopic_model(model):
    """Clean up BERTopic model internal data to prevent memory leaks."""
    try:
        # Clear UMAP embedding
        if hasattr(model, 'umap_model') and hasattr(model.umap_model, 'embedding_'):
            model.umap_model.embedding_ = None
        
        # Clear HDBSCAN labels and other data
        if hasattr(model, 'hdbscan_model'):
            if hasattr(model.hdbscan_model, 'labels_'):
                model.hdbscan_model.labels_ = None
            if hasattr(model.hdbscan_model, 'cluster_persistence_'):
                model.hdbscan_model.cluster_persistence_ = None
            if hasattr(model.hdbscan_model, 'condensed_tree_'):
                model.hdbscan_model.condensed_tree_ = None
            if hasattr(model.hdbscan_model, 'minimum_spanning_tree_'):
                model.hdbscan_model.minimum_spanning_tree_ = None
        
        # Clear vectorizer internal data
        if hasattr(model, 'vectorizer_model'):
            if hasattr(model.vectorizer_model, 'vocabulary_'):
                model.vectorizer_model.vocabulary_ = None
            if hasattr(model.vectorizer_model, 'stop_words_'):
                model.vectorizer_model.stop_words_ = None
            if hasattr(model.vectorizer_model, 'idf_'):
                model.vectorizer_model.idf_ = None
        
        # Clear c-TF-IDF model data
        if hasattr(model, 'ctfidf_model'):
            if hasattr(model.ctfidf_model, 'idf_'):
                model.ctfidf_model.idf_ = None
            if hasattr(model.ctfidf_model, 'X_'):
                model.ctfidf_model.X_ = None
        
        # Clear topic data
        if hasattr(model, 'topics_'):
            model.topics_ = None
        if hasattr(model, 'probabilities_'):
            model.probabilities_ = None
        if hasattr(model, 'topic_embeddings_'):
            model.topic_embeddings_ = None
        if hasattr(model, 'topic_labels_'):
            model.topic_labels_ = None
            
        # Clear document data
        if hasattr(model, 'documents_'):
            model.documents_ = None
        if hasattr(model, 'embeddings_'):
            model.embeddings_ = None
            
    except Exception:
        pass  # Ignore cleanup errors

# ============================================================================
# Configuration
# ============================================================================

class OptimizationConfig:
    """Configuration for clustering optimization."""
    
    # Optimization settings
    @staticmethod
    def get_default_n_trials(dataset_size: int) -> int:
        """Get number of trials based on dataset size with memory optimization."""
        if dataset_size <= 5000:
            return 20  # Reduced from 30
        elif dataset_size <= 20000:
            return 40  # Reduced from 60
        elif dataset_size <= 30000:
            return 50  # Reduced from 80
        else:
            return 60  # Reduced from 100
    
    @staticmethod
    def get_default_timeout(dataset_size: int) -> Optional[int]:
        """Get timeout in minutes based on dataset size."""
        if dataset_size <= 10000:
            return None
        elif dataset_size <= 50000:
            return 120
        else:
            return 240
    
    # Distance metrics
    UMAP_METRICS = ["cosine"]
    HDBSCAN_METRICS = ["euclidean", "manhattan"]
    
    # Topic representation
    TOP_N_WORDS_RANGE = (10, 20)
    NGRAM_RANGES = [[1, 3]]
    MIN_SAMPLES_MULTIPLIER_RANGE = (0.5, 1.0)
    
    # Score weights
    @staticmethod
    def get_adaptive_weights(dataset_size: int) -> Dict[str, float]:
        """Get balanced weights for both metrics."""
        return {
            'cluster_shape': 0.50,      # Silhouette UMAP
            'clustering_quality': 0.50  # DBCV Basis
        }  
    
    # Parameter ranges
    @staticmethod
    def get_min_df_range(dataset_size: int) -> Tuple[int, int]:
        """Get min_df range based on dataset size."""
        min_val = 2
        max_val = max(2, min(50, dataset_size // 1000))
        return (min_val, max_val)
    
    @staticmethod
    def get_max_df_range(dataset_size: int) -> Tuple[float, float]:
        """Get max_df range based on dataset size."""
        min_val = int(0.15 * dataset_size)  
        max_val = int(0.95 * dataset_size) 
        return (min_val, max_val) 
    
    @staticmethod
    def get_min_cluster_size_range(dataset_size: int) -> Tuple[int, int]:
        """Get min_cluster_size range based on dataset size."""
        min_val = max(10, dataset_size // 1000)
        max_val = min(2000, dataset_size // 20)
        return (min_val, max_val)
    
    @staticmethod
    def get_n_neighbors_range(dataset_size: int) -> Tuple[int, int]:
        """Get n_neighbors range based on dataset size."""
        min_val = max(10, min(30, dataset_size // 200))
        max_val = min(100, max(50, dataset_size // 100))
        return (min_val, max_val)
    
    @staticmethod
    def get_n_components_range(dataset_size: int) -> Tuple[int, int]:
        """Get n_components range based on dataset size."""
        min_val = 5
        max_val = min(20, max(10, dataset_size // 10000))
        return (min_val, max_val)

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

# ============================================================================
# Data Management
# ============================================================================

def load_papers(category: str) -> List[Paper]:
    """Load preprocessed papers for a given arXiv category."""
    filepath = f"./preprocessed/{category}/papers.pkl"
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Preprocessed papers not found at {filepath}")


def load_text_embeddings(category: str) -> np.ndarray:
    """Load pre-computed SPECTER2 text embeddings with memory mapping."""
    filepath = f"./preprocessed/{category}/text_embeddings.npy"
    try:
        # Use memory mapping to avoid loading entire file into memory
        return np.load(filepath, mmap_mode='r')
    except FileNotFoundError:
        raise FileNotFoundError(f"Text embeddings not found at {filepath}")

# ============================================================================
# Parameter Suggestion Functions
# ============================================================================

def _suggest_clustering_parameters(trial: optuna.Trial, dataset_size: int) -> Tuple[int, int, str]:
    """Suggest HDBSCAN clustering parameters with data-size adaptive bounds."""
    # Data-size adaptive ranges
    min_cluster_size_range = OptimizationConfig.get_min_cluster_size_range(dataset_size)
    min_cluster_size = trial.suggest_int(
        "min_cluster_size", 
        *min_cluster_size_range
    )
    
    # Derive min_samples from an independent multiplier to avoid dynamic search space
    # This keeps TPE multivariate sampling effective while preserving the constraint
    min_samples_multiplier = trial.suggest_float(
        "min_samples_multiplier",
        OptimizationConfig.MIN_SAMPLES_MULTIPLIER_RANGE[0],
        OptimizationConfig.MIN_SAMPLES_MULTIPLIER_RANGE[1]
    )
    # Ensure at least 1 and not exceeding min_cluster_size
    min_samples = max(1, min(int(min_cluster_size * min_samples_multiplier), min_cluster_size))
    
    # HDBSCAN distance metric (validated compatible metrics only)
    hdbscan_metric = trial.suggest_categorical("hdbscan_metric", OptimizationConfig.HDBSCAN_METRICS)
    
    return min_cluster_size, min_samples, hdbscan_metric


def _suggest_vectorization_parameters(trial: optuna.Trial, dataset_size: int) -> Tuple[int, List[int], int, int]:
    """Suggest text vectorization parameters with data-size adaptive ranges."""
    # Topic representation
    top_n_words = trial.suggest_int("top_n_words", *OptimizationConfig.TOP_N_WORDS_RANGE)
    
    # N-gram configuration
    ngram_range = trial.suggest_categorical("ngram_range", OptimizationConfig.NGRAM_RANGES)
    
    # Data-size adaptive TF-IDF bounds
    min_df_range = OptimizationConfig.get_min_df_range(dataset_size)
    min_df = trial.suggest_int("min_df", *min_df_range)
    
    max_df_range = OptimizationConfig.get_max_df_range(dataset_size)
    max_df = trial.suggest_int("max_df", *max_df_range)
    
    return top_n_words, ngram_range, min_df, max_df


def _suggest_umap_parameters(trial: optuna.Trial, dataset_size: int) -> Tuple[int, int, str]:
    """Suggest UMAP dimensionality reduction parameters with data-size adaptive ranges."""
    # Data-size adaptive ranges
    n_neighbors_range = OptimizationConfig.get_n_neighbors_range(dataset_size)
    n_neighbors = trial.suggest_int("n_neighbors", *n_neighbors_range)
    
    n_components_range = OptimizationConfig.get_n_components_range(dataset_size)
    n_components = trial.suggest_int("n_components", *n_components_range)
    
    # UMAP distance metric (optimized for SPECTER2 embeddings)
    umap_metric = trial.suggest_categorical("umap_metric", OptimizationConfig.UMAP_METRICS)
    
    return n_neighbors, n_components, umap_metric


def suggest_optimal_hyperparameters(trial: optuna.Trial, dataset_size: int) -> Hyperparameters:
    """Suggest complete hyperparameter set with simple fixed constraints."""
    min_cluster_size, min_samples, hdbscan_metric = _suggest_clustering_parameters(trial, dataset_size)
    top_n_words, ngram_range, min_df, max_df = _suggest_vectorization_parameters(trial, dataset_size)
    n_neighbors, n_components, umap_metric = _suggest_umap_parameters(trial, dataset_size)
    
    return Hyperparameters(
        top_n_words=top_n_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        n_neighbors=n_neighbors,
        n_components=n_components,
        umap_metric=umap_metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        hdbscan_metric=hdbscan_metric
    )

# ============================================================================
# Model Creation
# ============================================================================

def create_bertopic_model(params: Hyperparameters, embedding_model: CustomEmbeddingModel) -> BERTopic:
    """Create BERTopic model with optimized parameter configuration."""
    vectorizer_model = CountVectorizer(
        stop_words="english",
        analyzer="word",
        ngram_range=tuple(params.ngram_range),
        min_df=params.min_df,
        max_df=params.max_df,
        lowercase=False,  # Keep proper nouns for academic papers
        strip_accents="unicode"
    )
    
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
    
    umap_model = UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        metric=params.umap_metric,
        random_state=42,
        low_memory=True,  # Enable low memory mode
        #n_jobs=1  # Single thread to reduce memory usage
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric=params.hdbscan_metric,
        prediction_data=False,  # Disable prediction data to save memory
        #core_dist_n_jobs=1  # Single thread to reduce memory usage
    )
    
    return BERTopic(
        # nr_topics="auto",
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        top_n_words=params.top_n_words,
        calculate_probabilities=False,
        verbose=False
    )


# ============================================================================
# Evaluation Metrics
# ============================================================================

def _compute_silhouette_umap_score(
    model: BERTopic,
    original_embeddings: np.ndarray
) -> float:
    """Compute silhouette score using UMAP embedding."""
    try:
        labels = model.hdbscan_model.labels_
        
        # Filter out noise points (-1 labels)
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            return 0.0  # Not enough valid points
            
        valid_labels = labels[valid_mask]
        
        # Get UMAP embedding from the model
        umap_embedding = model.umap_model.embedding_
        if umap_embedding is None:
            return 0.0  # UMAP embedding not available
            
        valid_umap_embedding = umap_embedding[valid_mask]
        
        # Check if we have multiple clusters
        unique_labels = np.unique(valid_labels)
        if len(unique_labels) < 2:
            return 0.0  # Need at least 2 clusters for silhouette score
        
        # Compute silhouette score using euclidean metric on UMAP embeddings
        # Euclidean distance is optimal for low-dimensional embeddings (5-20 dimensions)
        silhouette_avg = silhouette_score(
            valid_umap_embedding, 
            valid_labels, 
            metric='euclidean'
        )
        
        # Normalize to [0, 1] range
        silhouette_score_normalized = (silhouette_avg + 1) / 2
        
        return silhouette_score_normalized
        
    except KeyboardInterrupt:
        raise    
    except Exception as e:
        print(f"Warning: Silhouette UMAP score computation failed: {e}")
        return 0.0  # Return neutral score on error


def _compute_dbcv_basis_score(
    model: BERTopic,
    original_embeddings: np.ndarray
) -> float:
    """Compute DBCV score using PCA with memory optimization."""
    try:
        labels = model.hdbscan_model.labels_
        
        # Filter out noise points (-1 labels)
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            return 0.0  # Not enough valid points
            
        valid_labels = labels[valid_mask]
        # Create a copy only if needed (for memory-mapped arrays)
        if hasattr(original_embeddings, 'base'):  # memory-mapped array
            valid_embeddings = np.array(original_embeddings[valid_mask])
        else:
            valid_embeddings = original_embeddings[valid_mask]
        
        # Check if we have multiple clusters
        unique_labels = np.unique(valid_labels)
        if len(unique_labels) < 2:
            return 0.0  # Need at least 2 clusters for DBCV
        
        # Memory-efficient PCA: Use fewer components for large datasets
        dataset_size = len(valid_embeddings)
        if dataset_size > 20000:
            pca = PCA(n_components=0.95, random_state=42)
        else:
            pca = PCA(n_components=0.99, random_state=42)
        
        projected_embeddings = pca.fit_transform(valid_embeddings)
        
        # Ensure float64 for HDBSCAN compatibility
        if projected_embeddings.dtype != np.float64:
            projected_embeddings = projected_embeddings.astype(np.float64)
        
        # Compute DBCV using cosine metric
        dbcv_score = validity_index(projected_embeddings, valid_labels, metric='cosine')
        
        # Normalize to [0, 1] range
        dbcv_score_normalized = (dbcv_score + 1) / 2
        
        # Clean up large arrays immediately
        del projected_embeddings, valid_embeddings, pca
        force_memory_cleanup()
        
        return dbcv_score_normalized
        
    except KeyboardInterrupt:
        raise    
    except Exception as e:
        print(f"Warning: DBCV basis score computation failed: {e}")
        return 0.0  # Return neutral score on error


def _get_basic_model_info(model: BERTopic, dataset_size: int) -> dict:
    """Get basic information about the trained model."""
    try:
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]
        n_topics = len(valid_topics)
        
        if n_topics > 0:
            cluster_sizes = valid_topics['Count'].values
            top_sizes = np.sort(cluster_sizes)[::-1][:3]  # Top 3 largest clusters
        else:
            top_sizes = []
        
        return {'n_topics': n_topics, 'top_cluster_sizes': top_sizes}
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt to be caught by outer try-except
        raise    
    except Exception as e:
        print(f"Warning: Failed to get basic model info: {e}")
        return {'n_topics': 0, 'top_cluster_sizes': []}

def compute_cluster_quality_score(
    model: BERTopic, 
    original_embeddings: np.ndarray,
    documents: List[str] = None
) -> float:
    """Compute combined clustering quality score with DBCV-based cluster shape analysis."""
    try:
        dataset_size = len(original_embeddings)
        basic_info = _get_basic_model_info(model, dataset_size)
        
        # Get adaptive weights first to determine which metrics to compute
        weights = OptimizationConfig.get_adaptive_weights(dataset_size)
        
        # Compute individual metrics only if their weights are non-zero
        silhouette_umap_score = 0.0
        dbcv_basis_score = 0.0
        
        if weights['cluster_shape'] > 0.00:
            silhouette_umap_score = _compute_silhouette_umap_score(model, original_embeddings)
        
        if weights['clustering_quality'] > 0.00:
            dbcv_basis_score = _compute_dbcv_basis_score(model, original_embeddings)
        
        final_score = (
            weights['cluster_shape'] * silhouette_umap_score +
            weights['clustering_quality'] * dbcv_basis_score
        )
        
        # Output results
        print(f"Topics: {basic_info['n_topics']}, Top sizes: {basic_info['top_cluster_sizes']}")
        
        # Build score and weight strings dynamically
        score_parts = []
        weight_parts = []
            
        if weights['cluster_shape'] > 0:
            score_parts.append(f"Silhouette UMAP: {silhouette_umap_score:.4f}")
            weight_parts.append(f"Silhouette UMAP: {weights['cluster_shape']:.1%}")
        else:
            score_parts.append("Silhouette UMAP: N/A")
            weight_parts.append("Silhouette UMAP: 0.0%")
            
        if weights['clustering_quality'] > 0:
            score_parts.append(f"DBCV Basis: {dbcv_basis_score:.4f}")
            weight_parts.append(f"DBCV: {weights['clustering_quality']:.1%}")
        else:
            score_parts.append("DBCV Basis: N/A")
            weight_parts.append("DBCV: 0.0%")
        
        print(f"Scores - {', '.join(score_parts)}")
        print(f"Weights - {', '.join(weight_parts)}")
        print(f"Final Score: {final_score:.4f}")
        print("-" * 60)
        
        return final_score
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt to be caught by outer try-except
        raise
    except Exception as e:
        print(f"Error in compute_cluster_quality_score: {e}")
        return 0.0  


# ============================================================================
# Optuna Configuration
# ============================================================================

def create_tpe_sampler(n_trials: int = 100, dataset_size: int = 10000) -> TPESampler:
    """Create TPE sampler optimized for SPECTER2-based academic paper clustering.
    
    Optimized for BERTopic with SPECTER2 embeddings and academic paper characteristics:
    - Dynamic startup trials: 15-20% of total trials for thorough exploration
    - Dynamic EI candidates: Scale with dataset size and trial count
    - multivariate=True: Essential for UMAP/HDBSCAN parameter correlations
    - Balanced prior weight: Optimized for academic paper clustering
    - group=False: Avoid grouping issues with mixed parameter types
    
    Args:
        n_trials: Total number of trials for optimization
        dataset_size: Size of the dataset for adaptive parameters
    """
    # Dynamic startup trials: 15-20% of total trials, min 10, max 20 (reduced for memory)
    # Academic papers need more exploration due to complex topic structures
    startup_trials = max(10, min(20, int(n_trials * 0.15)))
    
    # Dynamic EI candidates: Scale with dataset size and trials (reduced for memory)
    # Larger datasets need more candidates for better exploration
    if dataset_size <= 10000:
        ei_candidates = max(16, min(32, int(n_trials * 0.20)))
    elif dataset_size <= 50000:
        ei_candidates = max(20, min(40, int(n_trials * 0.25)))
    else:
        ei_candidates = max(24, min(48, int(n_trials * 0.30)))
    
    # Prior weight: Balanced for academic paper clustering
    # Slightly higher weight for better exploration of complex topic spaces
    prior_weight = 1.0
    
    return TPESampler(
        n_startup_trials=startup_trials,
        n_ei_candidates=ei_candidates,
        multivariate=True,  # Essential for UMAP/HDBSCAN correlations
        group=False,        # Avoid grouping issues
        prior_weight=prior_weight,
        warn_independent_sampling=True,
        seed=42
    )


def create_median_pruner(n_trials: int = 100, dataset_size: int = 10000) -> MedianPruner:
    """Create median pruner optimized for SPECTER2-based academic paper clustering.
    
    Optimized for BERTopic with SPECTER2 embeddings and academic paper characteristics:
    - Dynamic startup trials: 15-20% of total trials for thorough exploration
    - Extended warmup steps: Account for BERTopic's computation time and SPECTER2 embeddings
    - Conservative pruning: Academic papers have complex topic structures requiring patience
    - Adaptive intervals: Scale with dataset size for optimal pruning frequency
    
    Args:
        n_trials: Total number of trials for optimization
        dataset_size: Size of the dataset for adaptive parameters
    """
    # Dynamic startup trials: 15-20% of total trials, min 10, max 20 (reduced for memory)
    # Academic papers need more exploration due to complex topic structures
    startup_trials = max(10, min(20, int(n_trials * 0.15)))
    
    # Dynamic warmup steps: Scale with dataset size (reduced for memory)
    # Larger datasets take longer to converge, need more patience
    if dataset_size <= 10000:
        warmup_steps = 3
    elif dataset_size <= 50000:
        warmup_steps = 5
    else:
        warmup_steps = 7
    
    # Dynamic intervals: Scale with dataset size (more aggressive pruning)
    # Larger datasets need more frequent pruning checks
    if dataset_size <= 10000:
        interval_steps = 2
    elif dataset_size <= 50000:
        interval_steps = 1
    else:
        interval_steps = 1
    
    return MedianPruner(
        n_startup_trials=startup_trials,
        n_warmup_steps=warmup_steps,
        interval_steps=interval_steps
    )


def objective_function(
    trial: optuna.Trial, 
    texts: List[str], 
    text_embeddings: np.ndarray, 
    embedding_model: CustomEmbeddingModel
) -> float:
    """Optuna objective function for hyperparameter optimization."""
    dataset_size = len(texts)
    model = None
    
    try:
        # Suggest constrained hyperparameters
        params = suggest_optimal_hyperparameters(trial, dataset_size)
        
        # Create and train model
        model = create_bertopic_model(params, embedding_model)
        
        # For memory-mapped embeddings, create a copy only for this trial
        if hasattr(text_embeddings, 'base'):  # memory-mapped array
            embeddings_copy = np.array(text_embeddings)
        else:
            embeddings_copy = text_embeddings
            
        topics, _ = model.fit_transform(texts, embeddings=embeddings_copy)
        
        # Evaluate clustering quality
        score = compute_cluster_quality_score(model, embeddings_copy, documents=texts)
        
        # Store evaluation metrics
        trial.set_user_attr("score", float(score))
        
        # Clean up embeddings copy immediately
        del embeddings_copy
        force_memory_cleanup()
        
        return score
    
    except optuna.exceptions.TrialPruned:
        raise
    except KeyboardInterrupt:
        raise    
    except Exception as e:
        print(f"Warning: Trial failed: {e}")
        trial.set_user_attr("error", str(e))
        return 0.0
    finally:
        # Clean up model and force memory cleanup
        if model is not None:
            cleanup_bertopic_model(model)
            del model
        force_memory_cleanup()  


# ============================================================================
# Embedding Model Management
# ============================================================================

def create_embedding_model_with_cleanup():
    """Create embedding model with proper memory management."""
    try:
        model = get_custom_embedding_model()
        return model
    except Exception as e:
        print(f"Warning: Failed to create embedding model: {e}")
        return None

def cleanup_embedding_model(model):
    """Clean up embedding model to free GPU memory."""
    if model is not None:
        try:
            # Clear model from GPU memory
            if hasattr(model, 'model'):
                del model.model
            if hasattr(model, 'tokenizer'):
                del model.tokenizer
            # Force GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

# ============================================================================
# Main Optimization Pipeline
# ============================================================================

def optimize_category_clustering(
    category: str, 
    timeout: Optional[int] = None, 
    n_trials: Optional[int] = None,
    storage: Optional[str] = None
) -> optuna.Study:
    """Run hyperparameter optimization for a specific arXiv category with adaptive settings."""
    
    # Load and prepare data
    print(f"📂 Loading data for category: {category}")
    log_memory_usage("Before data loading")
    
    papers = load_papers(category)
    text_embeddings = load_text_embeddings(category)  # Now uses memory mapping
    embedding_model = create_embedding_model_with_cleanup()  # Create fresh instance per category
    
    if embedding_model is None:
        print(f"❌ Failed to create embedding model for category: {category}")
        return None
    
    texts = [embedding_model.get_input_text(paper) for paper in papers]
    dataset_size = len(texts)

    # Enhanced memory management
    del papers
    force_memory_cleanup()
    log_memory_usage("After data loading")
    
    # Enhanced memory safety check with dynamic limits
    memory_info = log_memory_usage("After data loading", verbose=False)
    available_memory = memory_info['available_mb']
    # recommended_limit = recommend_dataset_limit(available_memory)
    
    # if dataset_size > recommended_limit:
    #     print(f"⚠️  Dataset size exceeds memory limit for category: {category}")
    #     print(f"   • Dataset size: {dataset_size:,} documents")
    #     print(f"   • Recommended limit: {recommended_limit:,} documents")
    #     print(f"   • Available memory: {available_memory:.1f} MB")
    #     print(f"💡 Consider using a subset or increasing system memory")
    #     return None
    
    # Estimate memory requirements
    memory_estimate = get_dataset_memory_estimate(dataset_size)
    print(f"📊 Memory estimate: {memory_estimate['total_estimated_mb']:.1f} MB")
    
    print(f"📊 Dataset size: {dataset_size:,} documents")
    print(f"🧠 Using SPECTER2 Proximity adapter (110M parameters)")
    
    # Adaptive configuration based on dataset size
    if n_trials is None:
        n_trials = OptimizationConfig.get_default_n_trials(dataset_size)
    if timeout is None:
        timeout_minutes = OptimizationConfig.get_default_timeout(dataset_size)
        timeout = timeout_minutes * 60 if timeout_minutes else None  # Convert to seconds
    
    print(f"⚙️  Optimization settings:")
    print(f"   • Trials: {n_trials}")
    print(f"   • Timeout: {timeout//60 if timeout else 'None'} minutes")
    print(f"   • Sampler: TPE (multivariate=True)")
    print(f"   • Pruner: MedianPruner (adaptive)")
    
    # Create optimization study with adaptive sampler and pruner
    study_name = f"clustering_optimization_{category}_{dataset_size}"
    
    study = optuna.create_study(
        storage=storage,
        load_if_exists=True,  
        direction="maximize",
        study_name=study_name,
        sampler=create_tpe_sampler(n_trials, dataset_size),
        pruner=create_median_pruner(n_trials, dataset_size)
    )
    
    # Run optimization
    print(f"\n🚀 Starting optimization with {n_trials} trials...")
    print(f"📈 Progress will be shown below:")
    print("-" * 60)
    
    try:
        study.optimize(
            lambda trial: objective_function(trial, texts, text_embeddings, embedding_model),
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=True,
            catch=(ValueError, RuntimeError, MemoryError),
            callbacks=[
                # Enhanced memory cleanup callback every 5 trials
                lambda study, trial: (
                    force_memory_cleanup() if trial.number % 5 == 0 and trial.number > 0 else None
                )
            ]
        )
        
        # Display results
        _display_optimization_results(study)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Optimization interrupted. Completed {len(study.trials)} trials.")
        raise
    except Exception as e:
        print(f"❌ Optimization error: {e}")
    finally:
        # Enhanced cleanup after optimization
        cleanup_embedding_model(embedding_model)
        del texts, text_embeddings, embedding_model
        force_memory_cleanup()
        log_memory_usage("After optimization cleanup")
    
    return study


def _display_optimization_results(study: optuna.Study) -> None:
    """Display comprehensive optimization results."""
    if len(study.trials) == 0:
        print("❌ No completed trials found.")
        return
    
    best_trial = study.best_trial
    pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    success_rate = (len(study.trials) - pruned_count) / len(study.trials)
    
    print(f"\n{'='*60}")
    print(f"🎯 OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"🏆 Best score: {best_trial.value:.4f}")
    print(f"📊 Total trials: {len(study.trials)}")
    print(f"✂️  Pruned trials: {pruned_count}")
    print(f"✅ Success rate: {success_rate:.1%}")
    print(f"\n🔧 Best parameters:")
    for key, value in best_trial.params.items():
        print(f"   • {key}: {value}")
    print(f"{'='*60}")


def save_optimization_results(study: optuna.Study, output_dir: str) -> None:
    """Save optimization results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best parameters
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"💾 Best parameters saved to: {best_params_path}")
    print(f"📊 Study database: {output_dir}/search_params.db")


# ============================================================================
# Main Execution
# ============================================================================

def process_one_category(category: str):
    """Main execution function for hyperparameter optimization."""
    try:
        # Create output directory
        params_path = f"./params/{category}"
        os.makedirs(params_path, exist_ok=True)
        
        # Run optimization
        study_storage_path = f"sqlite:///{params_path}/search_params.db"
        study = optimize_category_clustering(
            category=category,
            storage=study_storage_path
        )
        
        # Save results
        save_optimization_results(study, params_path)
        
    finally:
        # Simple cleanup after each category
        force_memory_cleanup()


if __name__ == "__main__":
    # TODO: use logging
    # TODO: better memory cleanup
    # TODO: better exception handling especially for keyboard interrupt
    # TODO: abstruct score metrics
    # TODO: external config
    # TODO: save embeddings to DB
    # TODO: save params to DB
    # TODO: load papers from DB
    # TODO: save results to DB

    print("=" * 80)
    print("🔬 SPECTER2-BASED ACADEMIC PAPER CLUSTERING OPTIMIZATION")
    print("=" * 80)
    
    categories = get_category_codes()
    print(f"📚 Processing {len(categories)} arXiv categories:")
    for i, category in enumerate(categories, 1):
        print(f"  {i:2d}. {category}")
    
    print(f"\n🚀 Starting hyperparameter optimization...")
    print(f"📊 Using SPECTER2 Proximity adapter for academic paper clustering")
    print(f"🎯 Target: Optimize BERTopic parameters for topic discovery")
    print(f"⚙️  Pipeline: SPECTER2 → UMAP → HDBSCAN → Topic Modeling")
    print("-" * 80)
    
    try:
        for i, category in enumerate(categories, 1):
            print(f"\n📖 [{i}/{len(categories)}] Processing category: {category}")
            print(f"⏰ Started at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log_memory_usage(f"Before category {category}")
            
            try:
                process_one_category(category)
                print(f"✅ Completed category: {category}")
            except KeyboardInterrupt:
                # Re-raise KeyboardInterrupt to be caught by outer try-except
                raise
            except Exception as e:
                print(f"❌ Failed category {category}: {e}")
                continue
                
            # Simple cleanup between categories
            force_memory_cleanup()
            
    except KeyboardInterrupt:
        print(f"\n⚠️  INTERRUPTED BY USER (Ctrl+C)")
        print(f"🛑 Stopping optimization process...")
        print(f"📊 Processed {i-1}/{len(categories)} categories before interruption")
        print(f"💾 Partial results saved to: ./params/")
        print(f"🔄 To resume, run the script again (it will continue from where it left off)")
        exit(0)
    
    print("\n" + "=" * 80)
    print("🎉 ALL CATEGORIES PROCESSED SUCCESSFULLY!")
    print("📁 Results saved to: ./params/{category}/best_params.json")
    print("💾 Study data saved to: ./params/{category}/search_params.db")
    print("=" * 80)