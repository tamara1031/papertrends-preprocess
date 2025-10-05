from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import gc
import os
import pickle
import json
import numpy as np
import warnings

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
from hdbscan.validity import validity_index
from sklearn.metrics.pairwise import cosine_similarity

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
    """Centralized configuration for clustering optimization with simple fixed ranges."""
    
    # Numerical precision
    EPSILON = 1e-6
    
    # Optimization sessions
    DEFAULT_TIMEOUT = None  # No timeout by default
    
    # Distance metrics (validated for SPECTER2 -> UMAP -> HDBSCAN pipeline)
    UMAP_METRICS = ["cosine"]
    HDBSCAN_METRICS = ["euclidean", "manhattan"]
    
    # Fixed parameter ranges (simple and safe)
    TOP_N_WORDS_RANGE = (10, 30)
    NGRAM_RANGES = [[1, 2], [1, 3]]
    
    # Clustering parameters (fixed safe ranges)
    MIN_CLUSTER_SIZE_RANGE = (50, 500)
    MIN_SAMPLES_MAX_MULTIPLIER = 0.8
    
    # Vectorization parameters
    MIN_DF_RANGE = (2, 30)
    MAX_DF_PERCENT_RANGE = (0.1, 0.95)    # (10% to 95%)
    
    # UMAP parameters (fixed safe ranges)
    N_NEIGHBORS_RANGE = (10, 50)
    N_COMPONENTS_RANGE = (5, 15)

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
    """Suggest HDBSCAN clustering parameters with fixed safe bounds."""
    # Simple fixed ranges
    min_cluster_size = trial.suggest_int(
        "min_cluster_size", 
        *OptimizationConfig.MIN_CLUSTER_SIZE_RANGE
    )
    
    # min_samples constraint (relative to min_cluster_size)
    min_samples_max = max(3, int(min_cluster_size * OptimizationConfig.MIN_SAMPLES_MAX_MULTIPLIER))
    min_samples = trial.suggest_int("min_samples", 3, min_samples_max)
    
    # HDBSCAN distance metric (validated compatible metrics only)
    hdbscan_metric = trial.suggest_categorical("hdbscan_metric", OptimizationConfig.HDBSCAN_METRICS)
    
    return min_cluster_size, min_samples, hdbscan_metric


def _suggest_vectorization_parameters(trial: optuna.Trial, dataset_size: int) -> Tuple[int, List[int], int, int]:
    """Suggest text vectorization parameters with simple percentage ranges."""
    # Topic representation
    top_n_words = trial.suggest_int("top_n_words", *OptimizationConfig.TOP_N_WORDS_RANGE)
    
    # N-gram configuration
    ngram_range = trial.suggest_categorical("ngram_range", OptimizationConfig.NGRAM_RANGES)
    
    # Simple percentage-based TF-IDF bounds
    min_df = trial.suggest_int("min_df", *OptimizationConfig.MIN_DF_RANGE)
    max_df_percent = trial.suggest_float("max_df_percent", *OptimizationConfig.MAX_DF_PERCENT_RANGE)
    
    # Convert to integer counts (simple approach)
    max_df = int(max_df_percent * dataset_size)
    
    return top_n_words, ngram_range, min_df, max_df


def _suggest_umap_parameters(trial: optuna.Trial, dataset_size: int) -> Tuple[int, int, str]:
    """Suggest UMAP dimensionality reduction parameters with fixed ranges."""
    # Simple fixed ranges
    n_neighbors = trial.suggest_int("n_neighbors", *OptimizationConfig.N_NEIGHBORS_RANGE)
    n_components = trial.suggest_int("n_components", *OptimizationConfig.N_COMPONENTS_RANGE)
    
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
        low_memory=False
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


# ============================================================================
# Evaluation Metrics
# ============================================================================

def _compute_dbcv_score(
    model: BERTopic, 
    original_embeddings: np.ndarray, 
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute Density-Based Cluster Validation (DBCV) score."""
    labels = model.hdbscan_model.labels_
    
    # Remove outliers (-1 labels) for cleaner DBCV calculation
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return eps
    
    filtered_embeddings = original_embeddings[valid_mask].astype(np.float64)
    filtered_labels = labels[valid_mask]
    
    # Calculate distance matrix with enhanced numerical stability
    distance_matrix = pairwise_distances(filtered_embeddings, metric='euclidean').astype(np.float64)
    
    # Clean numeric issues (warnings suppressed globally)
    distance_matrix[distance_matrix <= 0] = eps
    distance_matrix[distance_matrix > 1e3] = 1e3
    distance_matrix[np.isnan(distance_matrix)] = eps
    distance_matrix[np.isinf(distance_matrix)] = 1e3
    
    # Compute DBCV score
    try:
        dbcv_score = validity_index(distance_matrix, filtered_labels)
    except Exception as e:
        print(f"Warning: DBCV computation failed: {e}")
        return eps
    
    # Normalize DBCV score from [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (dbcv_score + 1.0) / 2.0))


def _compute_topic_diversity(
    model: BERTopic, 
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute topic diversity based on cosine similarity of word vectors."""
    try:
        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]
        
        if len(valid_topics) < 2:
            return eps
        
        # Get all unique words across topics
        all_words = set()
        topic_word_lists = []
        
        for _, row in valid_topics.iterrows():
            topic_id = row['Topic']
            words = model.get_topic(topic_id)
            if words:
                word_list = [word[0] for word in words[:10]]
                topic_word_lists.append(word_list)
                all_words.update(word_list)
        
        if len(topic_word_lists) < 2:
            return eps
        
        # Create binary vectors for each topic
        all_words = list(all_words)
        topic_vectors = []
        
        for word_list in topic_word_lists:
            vector = [1 if word in word_list else 0 for word in all_words]
            topic_vectors.append(vector)
        
        # Calculate cosine similarity matrix
        topic_vectors = np.array(topic_vectors)
        similarity_matrix = cosine_similarity(topic_vectors)
        
        # Get upper triangle (excluding diagonal)
        similarities = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarities.append(similarity_matrix[i, j])
        
        if not similarities:
            return eps
        
        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(eps, min(1.0, diversity))
    
    except Exception as e:
        print(f"Warning: Topic diversity (cosine) computation failed: {e}")
        return eps


def _compute_topic_coverage(
    model: BERTopic, 
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute topic coverage - how well topics cover the documents."""
    try:
        labels = model.hdbscan_model.labels_
        total_docs = len(labels)
        
        if total_docs == 0:
            return eps
        
        # Count documents assigned to topics (excluding noise -1)
        assigned_docs = (labels != -1).sum()
        coverage = assigned_docs / total_docs
        
        return max(eps, min(1.0, coverage))
    
    except Exception as e:
        print(f"Warning: Topic coverage computation failed: {e}")
        return eps


def compute_cluster_quality_score(
    model: BERTopic, 
    original_embeddings: np.ndarray, 
    eps: float = OptimizationConfig.EPSILON
) -> float:
    """Compute combined clustering quality score with multiple metrics."""
    try:
        # Get number of topics (excluding noise topic -1)
        topic_info = model.get_topic_info()
        n_topics = len(topic_info[topic_info['Topic'] != -1])
        
        # Apply penalty if number of topics is 2 or fewer
        if n_topics <= 2:
            return 0.1  # Return low score for too few topics(not worse than eps)
        
        # Compute individual metrics
        dbcv_score = _compute_dbcv_score(model, original_embeddings, eps=eps)
        topic_diversity = _compute_topic_diversity(model, eps=eps)
        topic_coverage = _compute_topic_coverage(model, eps=eps)
        
        # Weighted combination of metrics
        # DBCV: 70% (density-based clustering quality including cohesion)
        # Topic Diversity: 20% (word overlap between topics)
        # Topic Coverage: 10% (document coverage)
        combined_score = (
            0.60 * dbcv_score +
            0.30 * topic_diversity +
            0.10 * topic_coverage
        )
        
        return combined_score
    except Exception:
        return eps


# ============================================================================
# Optuna Configuration
# ============================================================================

def create_tpe_sampler(study_name: str, dataset_size: int) -> TPESampler:
    """Create TPE sampler optimized for clustering hyperparameter search."""
    return TPESampler(
        consider_prior=True,
        prior_weight=0.85,
        consider_magic_clip=True,
        consider_endpoints=False,
        warn_independent_sampling=False,
        seed=42
    )


def create_median_pruner() -> MedianPruner:
    """Create median pruner for early stopping of poor trials."""
    return MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=3,
        interval_steps=3
    )


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
        score = compute_cluster_quality_score(model, text_embeddings, eps=eps)
        
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
    timeout: Optional[int] = None, 
    storage: Optional[str] = None
) -> optuna.Study:
    """Run hyperparameter optimization for a specific arXiv category."""
    timeout = timeout or OptimizationConfig.DEFAULT_TIMEOUT
    
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
        sampler=create_tpe_sampler(study_name, dataset_size),
        pruner=create_median_pruner()
    )
    
    # Run optimization
    n_trials = None
    
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

def main():
    """Main execution function for hyperparameter optimization."""
    category = "cs.AR"
    
    # Create output directory
    model_path = f"./models/{category}"
    os.makedirs(model_path, exist_ok=True)
    
    # Run optimization
    study_storage_path = f"sqlite:///{model_path}/search_params.db"
    study = optimize_category_clustering(
        category=category,
        storage=study_storage_path
    )
    
    # Save results
    save_optimization_results(study, model_path)


if __name__ == "__main__":
    main()