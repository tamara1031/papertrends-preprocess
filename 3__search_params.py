from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import gc
import os
import pickle
import json
import numpy as np
import warnings

import optuna
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import optuna.exceptions
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model, CustomEmbeddingModel

# Suppress expected numerical warnings (validated as safe)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='hdbscan.validity')
warnings.filterwarnings('ignore', message='overflow encountered in power')
warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

# ============================================================================
# Configuration
# ============================================================================

class OptimizationConfig:
    """Centralized configuration for clustering optimization with data-size adaptive ranges."""
    
    # Numerical precision
    EPSILON = 1e-6
    
    # Optimization sessions
    DEFAULT_TIMEOUT = None  # No timeout
    DEFAULT_N_TRIALS = 50
    
    # Distance metrics (validated for SPECTER2 -> UMAP -> HDBSCAN pipeline)
    UMAP_METRICS = ["cosine"]
    HDBSCAN_METRICS = ["euclidean", "manhattan"]  
    
    # Topic representation (data-size independent)
    TOP_N_WORDS_RANGE = (10, 20)
    NGRAM_RANGES = [[1, 3]]

    MIN_SAMPLES_MULTIPLIER_RANGE = (0.5, 1.0)  # Expanded range
    
    # Score weighting configuration (adaptive by dataset size)
    @staticmethod
    def get_adaptive_weights(dataset_size: int) -> Dict[str, float]:
        """Get adaptive weights based on dataset size for academic papers."""
        if dataset_size <= 5000:
            # Small datasets: focus on coverage and reasonable topic counts
            return {
                'coverage': 0.25,
                'topic_count': 0.15,
                'dominance': 0.15,
                'coherence': 0.25,
                'topic_diversity': 0.20
            }
        elif dataset_size <= 50000:
            # Medium datasets: balanced approach
            return {
                'coverage': 0.20,
                'topic_count': 0.10,
                'dominance': 0.15,
                'coherence': 0.30,
                'topic_diversity': 0.25
            }
        else:
            # Large datasets: focus on quality and diversity
            return {
                'coverage': 0.15,
                'topic_count': 0.05,
                'dominance': 0.15,
                'coherence': 0.35,
                'topic_diversity': 0.30
            }  
    
    # Data-size adaptive parameter ranges (optimized for 3K-200K documents)
    @staticmethod
    def get_min_df_range(dataset_size: int) -> Tuple[int, int]:
        """Get min_df range based on dataset size (conservative for academic abstracts)."""
        # More conservative ranges for academic abstracts
        min_val = max(2, dataset_size // 2000)  # 0.05% of dataset, min 2
        max_val = max(10, dataset_size // 500)   # 0.2% of dataset, min 10
        return (min_val, max_val)
    
    @staticmethod
    def get_max_df_range(dataset_size: int) -> Tuple[float, float]:
        """Get max_df range (conservative for academic abstracts)."""
        min_val = int(0.15 * dataset_size)  
        max_val = int(0.95 * dataset_size) 
        return (min_val, max_val) 
    
    @staticmethod
    def get_min_cluster_size_range(dataset_size: int) -> Tuple[int, int]:
        """Get min_cluster_size range based on dataset size (expanded for better exploration)."""
        # Expanded ranges for better optimization exploration
        min_val = max(10, dataset_size // 1000)   # 0.1% of dataset, min 10
        max_val = min(2000, dataset_size // 20)    # 5% of dataset, max 2000
        return (min_val, max_val)
    
    @staticmethod
    def get_n_neighbors_range(dataset_size: int) -> Tuple[int, int]:
        """Get n_neighbors range based on dataset size."""
        min_val = max(10, min(30, dataset_size // 200))  # Adaptive, min 10, max 30
        max_val = min(100, max(50, dataset_size // 100)) # Adaptive, min 50, max 100
        return (min_val, max_val)
    
    @staticmethod
    def get_n_components_range(dataset_size: int) -> Tuple[int, int]:
        """Get n_components range based on dataset size."""
        min_val = 5
        max_val = min(20, max(10, dataset_size // 10000))  # Adaptive, min 10, max 20
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
    """Load pre-computed SPECTER2 text embeddings."""
    filepath = f"./preprocessed/{category}/text_embeddings.npy"
    try:
        with open(filepath, "rb") as f:
            return np.load(f)
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
    
    # min_samples constraint (relative to min_cluster_size)
    min_samples_min = int(min_cluster_size * OptimizationConfig.MIN_SAMPLES_MULTIPLIER_RANGE[0])
    min_samples_max = int(min_cluster_size * OptimizationConfig.MIN_SAMPLES_MULTIPLIER_RANGE[1])
    min_samples = trial.suggest_int("min_samples", min_samples_min, min_samples_max)
    
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
        low_memory=False
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric=params.hdbscan_metric,
        prediction_data=True
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


def _compute_topic_coverage(
    model: BERTopic, 
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute topic coverage - how well topics cover the documents.
    
    Topic coverage measures the proportion of documents that are successfully
    assigned to meaningful topics (excluding noise/outliers). Higher coverage
    indicates better topic modeling performance with fewer unassigned documents.
    
    Args:
        model: Trained BERTopic model
        eps: Small epsilon for numerical stability
        
    Returns:
        Topic coverage score in range [0, 1] where 1 indicates 100% coverage
    """
    try:
        labels = model.hdbscan_model.labels_
        total_docs = len(labels)
        
        if total_docs == 0:
            return eps
        
        # Count documents assigned to topics (excluding noise -1)
        # HDBSCAN assigns -1 to noise/outlier points
        assigned_docs = (labels != -1).sum()
        
        # Calculate coverage ratio
        coverage = assigned_docs / total_docs
        
        # Ensure output is in valid range [0, 1]
        return max(eps, min(1.0, coverage))
    
    except Exception as e:
        print(f"Warning: Topic coverage computation failed: {e}")
        return eps


def _compute_dominance_score(
    model: BERTopic,
    dataset_size: int,
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute dominance score based on largest cluster size."""
    try:
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]
        cluster_sizes = valid_topics['Count'].values
        
        if len(cluster_sizes) == 0:
            return eps
            
        max_cluster_size = np.max(cluster_sizes)
        dominance_ratio = max_cluster_size / dataset_size
        # Smooth sigmoid-based penalty: starts penalizing around 0.3
        penalty_strength = 1.0 / (1.0 + np.exp(-10 * (dominance_ratio - 0.3)))
        return max(eps, 1.0 - penalty_strength)
        
    except Exception as e:
        print(f"Warning: dominance score computation failed: {e}")
        return eps

def _compute_topic_count_score(
    model: BERTopic,
    dataset_size: int,
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute topic count score based on optimal topic count for dataset size."""
    try:
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]
        n_topics = len(valid_topics)
        
        if n_topics <= 1:
            return eps
            
        # Optimal topic count based on dataset size
        # Use sqrt with logarithmic decay for natural progression
        base_count = np.sqrt(dataset_size)
        log_decay_factor = np.log10(max(10, dataset_size))
        
        # Calculate base topic count with coefficient 1.0
        base_topic_count = base_count * log_decay_factor
        
        # Define optimal coefficient range (0.04 to 0.08)
        optimal_min_coeff = 0.02
        optimal_max_coeff = 0.08
        
        # Calculate optimal topic count range
        optimal_min_topics = max(5, int(base_topic_count * optimal_min_coeff))
        optimal_max_topics = max(5, int(base_topic_count * optimal_max_coeff))
        
        # Use the midpoint of optimal range as the target
        optimal_topic_count = (optimal_min_topics + optimal_max_topics) // 2
        
        # Smooth optimization function: penalize deviation from optimal
        # Use Gaussian-like function centered at optimal_topic_count
        deviation_ratio = abs(n_topics - optimal_topic_count) / max(optimal_topic_count, 1)
        
        # Smooth penalty with adjustable width (sigma controls the width)
        sigma = 0.3  # Controls how quickly score drops from optimal
        return max(eps, np.exp(-0.5 * (deviation_ratio / sigma) ** 2))
        
    except Exception as e:
        print(f"Warning: topic count score computation failed: {e}")
        return eps

def _compute_topic_diversity_score(
    model: BERTopic,
    original_embeddings: np.ndarray,
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute topic diversity score using document embeddings with improved statistical robustness.
    
    This function computes topic diversity by:
    1. Computing centroids for topics with sufficient documents (>=2)
    2. Using median similarity for robustness against outliers
    3. Applying non-linear transformation for more natural diversity scoring
    4. Ensuring statistical validity through proper sample size requirements
    
    Args:
        model: Trained BERTopic model
        original_embeddings: Document embeddings used for clustering
        eps: Small epsilon for numerical stability
        
    Returns:
        Topic diversity score in range [eps, 1.0] where higher values indicate more diverse topics
    """
    try:
        # Get valid topics
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]
        
        if len(valid_topics) <= 1:
            return eps
            
        # Get document-topic assignments
        labels = model.hdbscan_model.labels_
        topic_centroids = []
        topic_sizes = []
        
        for topic_id in valid_topics['Topic'].values:
            # Get document indices for this topic
            topic_mask = (labels == topic_id)
            topic_indices = np.where(topic_mask)[0]
            
            # Require at least 2 documents for statistically meaningful centroid
            if len(topic_indices) >= 2:
                # Compute topic centroid from document embeddings
                topic_embeddings = original_embeddings[topic_indices]
                centroid = np.mean(topic_embeddings, axis=0)
                topic_centroids.append(centroid)
                topic_sizes.append(len(topic_indices))
        
        if len(topic_centroids) <= 1:
            return eps
            
        # Compute cosine similarities between topic centroids
        centroids_matrix = np.array(topic_centroids)
        similarities = cosine_similarity(centroids_matrix)
        
        # Get upper triangle similarities (excluding diagonal)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        
        if len(upper_triangle) == 0:
            return eps
            
        # Use median similarity for robustness against outliers
        median_similarity = np.median(upper_triangle)
        mean_similarity = np.mean(upper_triangle)
        
        # Apply non-linear transformation for more natural diversity scoring
        # Using exponential decay: diversity = exp(-similarity) 
        # This provides more intuitive scaling where:
        # - similarity=0 → diversity=1.0 (perfect diversity)
        # - similarity=1 → diversity≈0.37 (low diversity)
        # - similarity=0.5 → diversity≈0.61 (moderate diversity)
        diversity_score = np.exp(-median_similarity)
        
        # Additional penalty for high variance in similarities (indicates inconsistent diversity)
        similarity_std = np.std(upper_triangle)
        consistency_penalty = np.exp(-similarity_std * 2)  # Penalty for high variance
        diversity_score *= consistency_penalty
        
        print(f"Topic diversity - Topics: {len(topic_centroids)}, Median similarity: {median_similarity:.4f}, "
              f"Mean similarity: {mean_similarity:.4f}, Std: {similarity_std:.4f}, "
              f"Diversity: {diversity_score:.4f}")
        
        return max(eps, min(1.0, diversity_score))
        
    except Exception as e:
        print(f"Warning: Topic diversity computation failed: {e}")
        return eps


def _compute_embedding_coherence_score(
    model: BERTopic,
    original_embeddings: np.ndarray,
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute embedding-based coherence score using SPECTER2 document embeddings."""
    try:
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]
        
        if len(valid_topics) <= 1:
            return eps
            
        # Get document-topic assignments directly from HDBSCAN labels
        labels = model.hdbscan_model.labels_
        topic_coherences = []
        
        for topic_id in valid_topics['Topic'].values:
            # Get document indices for this topic
            topic_mask = (labels == topic_id)
            topic_indices = np.where(topic_mask)[0]
            
            if len(topic_indices) >= 2:
                topic_embeddings = original_embeddings[topic_indices]
                coherence = _compute_intra_topic_coherence(topic_embeddings, eps)
                if coherence > eps:
                    topic_coherences.append(coherence)
        
        # Apply non-linear transformation to emphasize high coherence scores
        mean_coherence = np.mean(topic_coherences) if topic_coherences else eps
        
        # Apply exponential transformation: exp(score) to create steeper slope at high values
        # This makes the score more sensitive to high coherence values
        transformed_score = np.exp(mean_coherence)
        
        return max(eps, min(1.0, transformed_score))

    except Exception:
        return eps

def _compute_intra_topic_coherence(
    topic_embeddings: np.ndarray, 
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute coherence for documents within a single topic using SPECTER2 embeddings."""
    try:
        if len(topic_embeddings) < 2:
            return eps
            
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(topic_embeddings)
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
        similarity_values = similarities[mask]
        
        if len(similarity_values) == 0:
            return eps
            
        avg_similarity = np.mean(similarity_values)
        return max(eps, min(1.0, avg_similarity))
        
    except Exception:
        return eps

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
        
    except Exception as e:
        print(f"Warning: Failed to get basic model info: {e}")
        return {'n_topics': 0, 'top_cluster_sizes': []}

def compute_cluster_quality_score(
    model: BERTopic, 
    original_embeddings: np.ndarray,
    documents: List[str] = None,
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute combined clustering quality score with cluster balance."""
    try:
        dataset_size = len(original_embeddings)
        basic_info = _get_basic_model_info(model, dataset_size)
        
        # Compute individual metrics
        topic_coverage = _compute_topic_coverage(model, eps=eps)
        dominance_score = _compute_dominance_score(model, dataset_size, eps=eps)
        topic_count_score = _compute_topic_count_score(model, dataset_size, eps=eps)
        topic_diversity_score = _compute_topic_diversity_score(model, original_embeddings, eps=eps)
        coherence_score = _compute_embedding_coherence_score(model, original_embeddings, eps=eps)
        
        # Get adaptive weights and compute final score
        weights = OptimizationConfig.get_adaptive_weights(dataset_size)
        final_score = (
            weights['coverage'] * topic_coverage +
            weights['dominance'] * dominance_score +
            weights['topic_count'] * topic_count_score +
            weights['topic_diversity'] * topic_diversity_score +
            weights['coherence'] * coherence_score
        )
        
        # Output results
        print(f"Topics: {basic_info['n_topics']}, Top sizes: {basic_info['top_cluster_sizes']}")
        print(f"Scores - Coverage: {topic_coverage:.4f}, Dominance: {dominance_score:.4f}, Topic Count: {topic_count_score:.4f}, Topic Diversity: {topic_diversity_score:.4f}, Embedding Coherence: {coherence_score:.4f}")
        print(f"Weights - Coverage: {weights['coverage']:.1%}, Dominance: {weights['dominance']:.1%}, Topic Count: {weights['topic_count']:.1%}, Topic Diversity: {weights['topic_diversity']:.1%}, Embedding Coherence: {weights['coherence']:.1%}")
        print(f"Final Score: {final_score:.4f}")
        print("-" * 60)
        
        return max(eps, min(1.0, final_score))
        
    except Exception as e:
        print(f"Error in compute_cluster_quality_score: {e}")
        return eps


# ============================================================================
# Optuna Configuration
# ============================================================================

def create_tpe_sampler(n_trials: int = 100) -> TPESampler:
    """Create TPE sampler optimized for BERTopic clustering hyperparameter search.
    
    Optimized for topic modeling with dynamic settings:
    - Dynamic startup trials: Align with pruner settings
    - Dynamic EI candidates: Scale with total trials
    - multivariate=True: Considers correlations between UMAP/HDBSCAN/TF-IDF parameters
    - Adjusted prior weight: Balanced exploration vs exploitation
    - group=False: Avoid complex grouping for BERTopic's mixed parameter types
    
    Args:
        study_name: Name of the optimization study
        dataset_size: Size of the dataset for adaptive parameters
        n_trials: Total number of trials for optimization
    """
    # Dynamic startup trials: Align with pruner (12% of total trials)
    # startup_trials = max(10, min(20, int(n_trials * 0.12)))
    
    # Dynamic EI candidates: Scale with total trials
    # ei_candidates = max(24, min(64, int(n_trials * 0.3)))  # 30% of trials, min 24, max 64
    
    return TPESampler(seed=42)


def create_median_pruner(n_trials: int = 100) -> MedianPruner:
    """Create median pruner optimized for BERTopic clustering optimization.
    
    Optimized settings for topic modeling with dynamic startup trials:
    - Dynamic startup trials: 10-15% of total trials, min 10, max 20
    - Extended warmup steps: Account for BERTopic's computation time
    - Reduced pruning frequency: Avoid premature pruning of promising trials
    
    Args:
        n_trials: Total number of trials for optimization
    """
    # Dynamic startup trials: 10-15% of total trials
    # startup_trials = max(10, min(20, int(n_trials * 0.12)))  # 12% as compromise
    
    return MedianPruner()


def objective_function(
    trial: optuna.Trial, 
    texts: List[str], 
    text_embeddings: np.ndarray, 
    embedding_model: CustomEmbeddingModel,
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Optuna objective function for hyperparameter optimization."""
    dataset_size = len(texts)
    
    try:
        # Suggest constrained hyperparameters
        params = suggest_optimal_hyperparameters(trial, dataset_size)
        
        # Create and train model
        model = create_bertopic_model(params, embedding_model)
        topics, _ = model.fit_transform(texts, embeddings=text_embeddings)
        
        # Evaluate clustering quality
        score = compute_cluster_quality_score(model, text_embeddings, documents=texts, eps=eps)
        
        # Store evaluation metrics
        trial.set_user_attr("score", float(score))
        
        return score
    
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Warning: Trial failed: {e}")
        trial.set_user_attr("error", str(e))
        return eps


# ============================================================================
# Main Optimization Pipeline
# ============================================================================

def optimize_category_clustering(
    category: str, 
    timeout: Optional[int] = OptimizationConfig.DEFAULT_TIMEOUT, 
    n_trials: Optional[int] = OptimizationConfig.DEFAULT_N_TRIALS,
    storage: Optional[str] = None
) -> optuna.Study:
    """Run hyperparameter optimization for a specific arXiv category."""
    
    # Load and prepare data
    print(f"Loading data for category: {category}")
    papers = load_papers(category)
    text_embeddings = load_text_embeddings(category)
    embedding_model = get_custom_embedding_model()
    
    texts = [embedding_model.get_input_text(paper) for paper in papers]
    dataset_size = len(texts)
    
    # Memory cleanup
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
        sampler=create_tpe_sampler(n_trials),
        pruner=create_median_pruner(n_trials)
    )
    
    # Run optimization
    print(f"Starting optimization with {n_trials} trials...")
    
    try:
        study.optimize(
            lambda trial: objective_function(trial, texts, text_embeddings, embedding_model),
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=True,
            catch=(Exception,)  # Catch all exceptions gracefully...
        )
        
        # Display results
        _display_optimization_results(study)
            
    except KeyboardInterrupt:
        print(f"\nOptimization interrupted. Completed {len(study.trials)} trials.")
    except Exception as e:
        print(f"Optimization error: {e}")
    
    return study


def _display_optimization_results(study: optuna.Study) -> None:
    """Display comprehensive optimization results."""
    if len(study.trials) == 0:
        print("No completed trials found.")
        return
    
    best_trial = study.best_trial
    pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    print(f"\n{'='*50}")
    print(f"OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    print(f"Best score: {best_trial.value:.4f}")
    print(f"Best parameters:")
    print(json.dumps(best_trial.params, indent=2))
    print(f"Total trials: {len(study.trials)}")
    print(f"Pruned trials: {pruned_count}")
    print(f"Success rate: {(len(study.trials) - pruned_count) / len(study.trials):.1%}")
    print(f"{'='*50}")


def save_optimization_results(study: optuna.Study, output_dir: str) -> None:
    """Save optimization results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best parameters
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"Best parameters saved to: {best_params_path}")


# ============================================================================
# Main Execution
# ============================================================================

def process_one_category(category: str):
    """Main execution function for hyperparameter optimization."""
    
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


if __name__ == "__main__":
    category = "cs.AR"
    process_one_category(category)