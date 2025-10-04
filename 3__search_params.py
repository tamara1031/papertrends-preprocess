"""
Hyperparameter Optimization for BERTopic Clustering

This module implements automated hyperparameter optimization for BERTopic clustering
using Optuna. It optimizes parameters for text vectorization, UMAP dimensionality
reduction, and HDBSCAN clustering with combined coherence and DBCV evaluation metrics.

Author: PaperTrends Preprocessing Team
"""

from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import gc
import os
import pickle
import json
import numpy as np
import warnings
from itertools import combinations

import optuna
import optuna.exceptions
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan.validity import validity_index

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model, CustomEmbeddingModel

# Suppress expected numerical warnings from HDBSCAN and related libraries
warnings.filterwarnings('ignore', category=RuntimeWarning, module='hdbscan.validity')
warnings.filterwarnings('ignore', message='overflow encountered in power')
warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

# ============================================================================
# Configuration Constants
# ============================================================================

class ClusteringConfig:
    """Configuration parameters for clustering optimization."""
    
    # Numerical stability
    EPSILON: float = 1e-6
    
    # Clustering constraints - these ratios determine min/max cluster sizes
    MIN_CLUSTER_RATIO: int = 20  # Min cluster size = dataset_size // MIN_CLUSTER_RATIO
    MAX_CLUSTER_RATIO: int = 500  # Max cluster size = dataset_size // MAX_CLUSTER_RATIO
    
    # UMAP dimensionality reduction parameters
    UMAP_NEIGHBORS_RATIO: float = 0.03  # Max n_neighbors = dataset_size * UMAP_NEIGHBORS_RATIO
    UMAP_MAX_NEIGHBORS: int = 50  # Absolute maximum for n_neighbors
    UMAP_MIN_COMPONENTS: int = 2
    UMAP_MAX_COMPONENTS: int = 15
    
    # Optimization settings
    DEFAULT_TIMEOUT: int | None = None
    DEFAULT_TRIALS: int = 100
    MIN_TRIALS: int = 30
    MAX_TRIALS: int = 100
    TRIALS_SCALE_FACTOR: int = 50  # Trials scale with dataset_size // TRIALS_SCALE_FACTOR


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Hyperparameters:
    """Hyperparameters for BERTopic clustering optimization."""

    top_n_words: int
    
    # Text vectorization parameters
    ngram_range: List[int]
    min_df: Union[float, int]
    max_df: Union[float, int]
    
    # UMAP dimensionality reduction parameters
    n_neighbors: int
    n_components: int
    umap_metric: str
    
    # HDBSCAN clustering parameters
    min_cluster_size: int
    min_samples: int
    hdbscan_metric: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hyperparameters to dictionary format."""
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
# Data Loading and Preparation
# ============================================================================

def load_papers(category: str) -> List[Paper]:
    """
    Load papers from preprocessed data for the given category.
    
    Args:
        category: The arXiv category (e.g., 'physics.geo-ph')
        
    Returns:
        List of Paper objects preprocessed for the category
        
    Raises:
        FileNotFoundError: If preprocessed data doesn't exist
    """
    filepath = f"./preprocessed/{category}/papers.pkl"
    try:
        with open(filepath, "rb") as f:
            papers = pickle.load(f)
        return papers
    except FileNotFoundError:
        raise FileNotFoundError(f"Preprocessed papers not found at {filepath}")


def load_text_embeddings(category: str) -> np.ndarray:
    """
    Load pre-computed text embeddings for the given category.
    
    Args:
        category: The arXiv category (e.g., 'physics.geo-ph')
        
    Returns:
        Numpy array of shape (n_samples, embedding_dim) containing embeddings
        
    Raises:
        FileNotFoundError: If embeddings file doesn't exist
    """
    filepath = f"./preprocessed/{category}/text_embeddings.npy"
    try:
        with open(filepath, "rb") as f:
            embeddings = np.load(f)
        return embeddings
    except FileNotFoundError:
        raise FileNotFoundError(f"Text embeddings not found at {filepath}")


# ============================================================================
# Evaluation Metrics
# ============================================================================

def _compute_word_coherence(words: List[str], model: BERTopic) -> float:
    """
    Compute topic coherence for a set of words using embedding similarity.
    
    This is a simplified coherence measure based on word embedding similarity.
    True topic coherence ideally requires external corpus analysis.
    
    Args:
        words: List of words for a specific topic
        model: Fitted BERTopic model for accessing embeddings
    
    Returns:
        Average pairwise cosine similarity between word embeddings [0, 1]
    """
    if len(words) < 2:
        return 0.0
    
    try:
        # Get word embeddings from the model
        embeddings = model.embedding_model.embed(words)
        
        # Optimize computation based on vocabulary size
        if len(words) <= 50:
            # For small vocabularies, compute pairwise similarities individually
            similarities = []
            for w1_idx, w2_idx in combinations(range(len(words)), 2):
                sim = cosine_similarity([embeddings[w1_idx]], [embeddings[w2_idx]])[0, 0]
                similarities.append(sim)
        else:
            # For large vocabularies, compute full similarity matrix and extract upper triangle
            similarity_matrix = cosine_similarity(embeddings)
            upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
            similarities = similarity_matrix[upper_triangle_indices]
        
        return np.mean(similarities) if similarities else 0.0
        
    except Exception as e:
        # Return 0.0 if embedding computation fails (rare edge case)
        print(f"Warning: Word coherence computation failed: {e}")
        return 0.0


def _compute_topic_coherence_score(model: BERTopic, eps: float = ClusteringConfig.EPSILON) -> float:
    """
    Calculate document-count weighted topic coherence score.
    
    Args:
        model: Fitted BERTopic model
        eps: Minimum score for degenerated cases
        
    Returns:
        Weighted average coherence score normalized to [0, 1]
    """
    # Extract topic words (excluding outlier topic -1)
    topic_words_dict = model.get_topics()
    topic_words_dict = {
        k: [word for word, _ in words_tuples] 
        for k, words_tuples in topic_words_dict.items() 
        if k != -1
    }
    
    if not topic_words_dict:
        return eps
    
    # Count documents per topic
    topic_counts = {}
    for label in model.hdbscan_model.labels_:
        if label != -1:  # Exclude outliers
            topic_counts[label] = topic_counts.get(label, 0) + 1
    
    # Compute coherence scores for each topic
    topic_coherences = []
    topic_weights = []
    
    for topic_id, words in topic_words_dict.items():
        if len(words) < 2:  # Skip topics with insufficient words
            continue
        
        topic_coherence = _compute_word_coherence(words, model)
        topic_coherences.append(topic_coherence)
        
        # Use document count as weight
        doc_count = topic_counts.get(topic_id, 1)
        topic_weights.append(doc_count)
    
    if not topic_coherences:
        return eps
    
    # Calculate document-count weighted average coherence
    topic_coherences = np.array(topic_coherences)
    topic_weights = np.array(topic_weights)
    
    weighted_avg_coherence = np.average(topic_coherences, weights=topic_weights)
    
    # Normalize from [-1, 1] to [0, 1] range
    normalized_coherence = (weighted_avg_coherence + 1.0) / 2.0
    
    return normalized_coherence


def _compute_dbcv_score(
    model: BERTopic, 
    original_embeddings: np.ndarray, 
    eps: float = ClusteringConfig.EPSILON
) -> float:
    """
    Compute Density-Based Cluster Validation (DBCV) score.
    
    Args:
        model: Fitted BERTopic model
        original_embeddings: Original document embeddings
        eps: Minimum score for degenerated cases
        
    Returns:
        Normalized DBCV score in [0, 1] range
    """
    labels = model.hdbscan_model.labels_
    
    # Remove outliers (-1 labels) for cleaner DBCV calculation
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return eps
    
    filtered_embeddings = original_embeddings[valid_mask].astype(np.float64)
    filtered_labels = labels[valid_mask]
    
    # Calculate distance matrix with numerical stability preprocessing
    distance_matrix = pairwise_distances(filtered_embeddings, metric='euclidean').astype(np.float64)
    
    # Numerical stability preprocessing for HDBSCAN validity
    distance_matrix[distance_matrix <= 0] = eps  # Avoid division by zero
    distance_matrix[distance_matrix > 1e3] = 1e3  # Conservative overflow prevention
    distance_matrix[np.isnan(distance_matrix)] = eps  # Handle NaN values
    distance_matrix[np.isinf(distance_matrix)] = 1e3  # Handle infinity values
    
    # Compute DBCV score (warnings already suppressed globally)
    try:
        dbcv_score = validity_index(distance_matrix, filtered_labels)
    except Exception as e:
        print(f"Warning: DBCV computation failed: {e}")
        return eps
    
    # Normalize DBCV score from [-1, 1] to [0, 1]
    normalized_score = max(0.0, min(1.0, (dbcv_score + 1.0) / 2.0))
    
    return normalized_score


def compute_cluster_quality_score(
    model: BERTopic, 
    original_embeddings: np.ndarray, 
    eps: float = ClusteringConfig.EPSILON
) -> float:
    """
    Compute combined clustering quality score from coherence and DBCV metrics.
    
    Args:
        model: Fitted BERTopic model
        original_embeddings: Original document embeddings  
        eps: Minimum score for degenerated cases
        
    Returns:
        Combined quality score weighted: 40% coherence + 60% DBCV
    """
    try:
        coherence_score = _compute_topic_coherence_score(model, eps=eps)
        dbcv_score = _compute_dbcv_score(model, original_embeddings, eps=eps)
        
        combined_score = 0.4 * coherence_score + 0.6 * dbcv_score
        
        return combined_score
    
    except Exception:
        return eps


# ============================================================================
# Optuna Configuration
# ============================================================================

def create_tpe_sampler(study_name: str, dataset_size: int) -> TPESampler:
    """
    Create TPE sampler optimized for clustering hyperparameter search.
    
    Args:
        study_name: Name of the Optuna study
        dataset_size: Number of documents in dataset
        
    Returns:
        Configured TPESampler instance
    """
    return TPESampler(
        consider_prior=True,           # Use Bayesian mixture for better adaptation
        prior_weight=0.85,            # Balance between exploration and exploitation
        consider_magic_clip=True,      # Adaptive clipping for extreme values
        consider_endpoints=False,      # Exclude boundary parameter values
        warn_independent_sampling=False,  # Suppress dynamic search space warnings
        seed=42                       # Reproducible random sampling
    )


def create_median_pruner() -> MedianPruner:
    """
    Create median pruner for early stopping of poor trials.
    
    Returns:
        Configured MedianPruner instance
    """
    return MedianPruner(
        n_startup_trials=10,    # Wait for sufficient trials before pruning
        n_warmup_steps=3,       # Warmup period without pruning
        interval_steps=3        # Check for pruning every 3 steps
    )


# ============================================================================
# Parameter Suggestion Functions
# ============================================================================

def _suggest_clustering_parameters(trial: optuna.Trial, dataset_size: int) -> Tuple[int, int, str]:
    """
    Suggest HDBSCAN clustering parameters with dataset-aware constraints.
    
    Args:
        trial: Optuna trial instance
        dataset_size: Number of documents in dataset
        
    Returns:
        Tuple of (min_cluster_size, min_samples, hdbscan_metric)
    """
    # Core clustering parameters optimized for practical cluster counts
    min_cluster_size_lower = max(5, dataset_size // ClusteringConfig.MAX_CLUSTER_RATIO)
    min_cluster_size_upper = min(100, dataset_size // ClusteringConfig.MIN_CLUSTER_RATIO)
    
    min_cluster_size = trial.suggest_int(
        "min_cluster_size", 
        min_cluster_size_lower, 
        min_cluster_size_upper
    )
    
    # min_samples should be constrained relative to min_cluster_size
    min_samples_max = max(3, int(min_cluster_size * 0.8))
    min_samples = trial.suggest_int("min_samples", 3, min_samples_max)
    
    # HDBSCAN distance metric options (limited to supported metrics)
    hdbscan_metric = trial.suggest_categorical(
        "hdbscan_metric", 
        ["euclidean", "manhattan"]
    )
    
    return min_cluster_size, min_samples, hdbscan_metric


def _suggest_vectorization_parameters(trial: optuna.Trial, dataset_size: int) -> Tuple[int, List[int], int, float]:
    """
    Suggest text vectorization parameters with mutual constraints.
    
    Args:
        trial: Optuna trial instance
        dataset_size: Number of documents in dataset
        
    Returns:
        Tuple of (top_n_words, ngram_range, min_df, max_df)
    """
    # Topic representation parameter
    top_n_words = trial.suggest_int("top_n_words", 10, 30)
    
    ngram_range = trial.suggest_categorical("ngram_range", [[1, 2], [1, 3]])
    
    # Use percentage-based approach for better constraint handling
    # Convert min_df from count to percentage first  
    min_df_percent_min = 2 / dataset_size  # 2 documents as percentage
    min_df_percent_max = min(0.01, 50 / dataset_size)  # At most 1% or 50 documents
    
    min_df_percent = trial.suggest_float("min_df_percent", min_df_percent_min, min_df_percent_max)
    min_df = max(2, int(min_df_percent * dataset_size))  # Round up to integer count
    
    # Now calculate max_df with proper constraints
    max_df_min = min_df_percent + 0.005  # At least 0.5% buffer above min_df_percent
    max_df_min_safe = max(max_df_min, 0.015)  # At least 1.5% minimum
    
    max_df_max = min(max_df_min_safe + 0.2, 0.9)  # At least 20% range, but not more than 90%
    
    max_df = trial.suggest_float("max_df", max_df_min_safe, max_df_max)
    
    return top_n_words, ngram_range, min_df, max_df


def _suggest_umap_parameters(trial: optuna.Trial, dataset_size: int) -> Tuple[int, int, str]:
    """
    Suggest UMAP dimensionality reduction parameters.
    
    Args:
        trial: Optuna trial instance
        dataset_size: Number of documents in dataset
        
    Returns:
        Tuple of (n_neighbors, n_components, umap_metric)
    """
    # Constrain n_neighbors based on dataset size
    practical_max = min(
        int(dataset_size * ClusteringConfig.UMAP_NEIGHBORS_RATIO), 
        ClusteringConfig.UMAP_MAX_NEIGHBORS
    )
    
    n_neighbors = trial.suggest_int("n_neighbors", 5, practical_max)
    
    # Constrain dimensionality based on dataset size
    min_components = max(
        ClusteringConfig.UMAP_MIN_COMPONENTS, 
        int(np.log10(dataset_size))
    )
    max_components = min(
        ClusteringConfig.UMAP_MAX_COMPONENTS, 
        dataset_size // 100
    )
    
    n_components = trial.suggest_int("n_components", min_components, max_components)
    
    # UMAP distance metric optimized for SPECTER2 embeddings
    # Cosine is preferred for semantic similarities, but allow optimization
    umap_metric_options = ["cosine", "euclidean", "manhattan"]
    umap_metric = trial.suggest_categorical("umap_metric", umap_metric_options)
    
    return n_neighbors, n_components, umap_metric


def suggest_optimal_hyperparameters(trial: optuna.Trial, dataset_size: int) -> Hyperparameters:
    """
    Suggest complete hyperparameter set with cross-parameter constraints.
    
    Args:
        trial: Optuna trial instance
        dataset_size: Number of documents in dataset
        
    Returns:
        Hyperparameters dataclass instance
    """
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
# Model Creation and Training
# ============================================================================

def create_bertopic_model(params: Hyperparameters, embedding_model: CustomEmbeddingModel) -> BERTopic:
    """
    Create BERTopic model with specified hyperparameters.
    
    Args:
        params: Hyperparameters for model configuration
        embedding_model: Custom embedding model instance
        
    Returns:
        Configured BERTopic model instance
    """
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=tuple(params.ngram_range),
        min_df=params.min_df,
        max_df=params.max_df,
        lowercase=False,
        strip_accents="unicode"
    )
    
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
    
    umap_model = UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        metric=params.umap_metric,
        random_state=42,
        low_memory=False  # Better performance for parameter search
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric=params.hdbscan_metric,
        prediction_data=True
    )
    
    return BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        top_n_words=params.top_n_words,
        calculate_probabilities=False,
        verbose=False
    )


def objective_function(
    trial: optuna.Trial, 
    texts: List[str], 
    text_embeddings: np.ndarray, 
    embedding_model: CustomEmbeddingModel,
    eps: float = ClusteringConfig.EPSILON
) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial instance
        texts: List of input text documents
        text_embeddings: Pre-computed text embeddings
        embedding_model: Custom embedding model
        eps: Minimum score for failed evaluations
        
    Returns:
        Combined clustering quality score
    """
    dataset_size = len(texts)
    
    try:
        # Suggest constrained hyperparameters
        params = suggest_optimal_hyperparameters(trial, dataset_size)
        
        # Create and train model
        model = create_bertopic_model(params, embedding_model)
        topics, _ = model.fit_transform(texts, embeddings=text_embeddings)
        
        # Validate clustering results
        topic_info = model.get_topic_info()
        n_clusters = len(topic_info[topic_info['Topic'] != -1])
        
        # Early termination if no valid clusters found
        if n_clusters == 0:
            trial.report(eps, 0)
            raise optuna.exceptions.TrialPruned("No valid clusters found")
        
        # Evaluate clustering quality
        combined_score = compute_cluster_quality_score(model, text_embeddings, eps=eps)
        
        # Store evaluation metrics for analysis
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("combined_score", float(combined_score))
        
        return combined_score
    
    except optuna.exceptions.TrialPruned:
        raise  # Re-raise pruned trials
    except Exception as e:
        # Graceful handling of other failures
        print(f"Warning: Trial failed: {e}")
        trial.set_user_attr("error", str(e))
        return eps


# ============================================================================
# Main Optimization Pipeline
# ============================================================================

def optimize_category_clustering(
    category: str, 
    timeout: Optional[int] = None, 
    storage: Optional[str] = None
) -> optuna.Study:
    """
    Run hyperparameter optimization for a specific category.
    
    Args:
        category: The arXiv category to optimize (e.g., 'physics.geo-ph')
        timeout: Maximum optimization time in seconds (default: 10 minutes)
        storage: Database URL for study persistence (optional)
        
    Returns:
        Completed Optuna study with optimization results
    """
    timeout = timeout or ClusteringConfig.DEFAULT_TIMEOUT
    
    # Load and prepare data
    print(f"Loading data for category: {category}")
    papers = load_papers(category)
    text_embeddings = load_text_embeddings(category)
    embedding_model = get_custom_embedding_model()
    
    texts = [embedding_model.get_input_text(paper) for paper in papers]
    dataset_size = len(texts)
    
    # Clean up paper objects to save memory
    del papers
    gc.collect()
    
    print(f"Dataset size: {dataset_size} documents")
    
    # Create optimization study
    study_name = f"clustering_optimization_{category}_{dataset_size}"
    
    study = optuna.create_study(
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        study_name=study_name,
        sampler=create_tpe_sampler(study_name, dataset_size),
        pruner=create_median_pruner()
    )
    
    # Run optimization
    n_trials = min(
        ClusteringConfig.MAX_TRIALS, 
        max(ClusteringConfig.MIN_TRIALS, dataset_size // ClusteringConfig.TRIALS_SCALE_FACTOR)
    )
    
    print(f"Starting optimization with {n_trials} trials...")
    
    try:
        study.optimize(
            lambda trial: objective_function(trial, texts, text_embeddings, embedding_model),
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=True,
            catch=(Exception,)  # Catch all exceptions gracefully
        )
        
        # Display optimization results
        if len(study.trials) > 0:
            best_trial = study.best_trial
            print(f"\n{'='*50}")
            print(f"OPTIMIZATION RESULTS")
            print(f"{'='*50}")
            print(f"Best score: {best_trial.value:.4f}")
            print(f"Best parameters: {json.dumps(best_trial.params, indent=2)}")
            print(f"Total trials: {len(study.trials)}")
            print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
            print(f"{'='*50}")
        else:
            print("No completed trials found.")
            
    except KeyboardInterrupt:
        print(f"\nOptimization interrupted. Completed {len(study.trials)} trials.")
    except Exception as e:
        print(f"Optimization error: {e}")
    
    return study


def save_optimization_results(study: optuna.Study, output_dir: str) -> None:
    """
    Save optimization results to disk.
    
    Args:
        study: Completed Optuna study
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best parameters
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"Best parameters saved to: {best_params_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    category = "physics.geo-ph"
    
    # Create output directory
    model_path = f"./models/{category}"
    os.makedirs(model_path, exist_ok=True)
    
    # Run optimization
    study_storage_path = f"sqlite:///{model_path}/search_params.db"
    study = optimize_category_clustering(
        category=category,
        # timeout=20 * 60,
        storage=study_storage_path
    )
    
    # Save results
    save_optimization_results(study, model_path)


if __name__ == "__main__":
    main()