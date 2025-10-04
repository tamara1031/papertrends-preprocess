from typing import List, Optional, Union
from dataclasses import dataclass
import gc
import os
import pickle
import json
import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model

# Constants
EPSILON = 1e-6
EMBEDDING_MODEL = get_custom_embedding_model()

# Clustering constraints
MIN_CLUSTERS = 2
MAX_CLUSTERS_RATIO = 5  # Max clusters = dataset_size // MAX_CLUSTERS_RATIO
MIN_CLUSTER_RATIO = 20  # Min cluster size = dataset_size // MIN_CLUSTER_RATIO
MAX_CLUSTER_RATIO = 500  # Max cluster size = dataset_size // MAX_CLUSTER_RATIO

# UMAP parameters
UMAP_NEIGHBORS_RATIO = 0.03  # Max n_neighbors = dataset_size * UMAP_NEIGHBORS_RATIO
UMAP_MAX_NEIGHBORS = 50  # Absolute maximum for n_neighbors
UMAP_MIN_COMPONENTS = 2
UMAP_MAX_COMPONENTS = 15

# Optimization limits
EARLY_PRUNING_THRESHOLD = 20
MID_OPTIMIZATION_THRESHOLD = 100

# Score weights for composite evaluation
SCORE_WEIGHTS = {
    'coherence': 0.40,      # Topic internal consistency
    'cluster': 0.40,        # Clustering quality metrics
    'validity': 0.20        # Practical cluster count appropriateness
}


def get_papers(category: str) -> List[Paper]:
    """Load papers from preprocessed data for the given category."""
    with open(f"./preprocessed/{category}/papers.pkl", "rb") as f:
        papers = pickle.load(f)
    return papers


def get_text_embeddings(category: str) -> np.ndarray:
    """Load pre-computed text embeddings for the given category."""
    with open(f"./preprocessed/{category}/text_embeddings.npy", "rb") as f:
        embeddings = np.load(f)
    return embeddings

@dataclass
class Hyperparameters:
    """
    Configuration parameters for BERTopic model optimization.
    
    This dataclass encapsulates all tunable hyperparameters for:
    - Text vectorization (ngram_range, min_df, max_df)
    - UMAP dimensionality reduction (n_neighbors, n_components, min_dist, spread)
    - HDBSCAN clustering (min_cluster_size, min_samples)
    """
    ngram_range: List[int]
    min_df: Union[float, int]
    max_df: Union[float, int]
    n_neighbors: int
    n_components: int
    min_dist: float
    spread: float
    min_cluster_size: int
    min_samples: int

def compute_coherence(model: BERTopic, top_n: int = 10, eps: float = EPSILON) -> float:
    """
    Compute topic coherence using cosine similarity between word embeddings.
    
    Args:
        model: Trained BERTopic model
        top_n: Number of top words per topic to consider
        eps: Minimum epsilon value
        
    Returns:
        Coherence score between 0 and 1
    """
    try:
        topics = {k: v for k, v in model.get_topics().items() if k != -1}
        if len(topics) < 2:
            return eps

        coherence_scores = []
        
        # Calculate coherence for each topic separately
        for topic_id, topic_words in topics.items():
            if len(topic_words) < 2:
                continue
                
            words = [word for word, _ in topic_words[:top_n]]
            if len(words) < 2:
                continue
            
            try:
                # Get word embeddings using the embedding model
                word_vectors = EMBEDDING_MODEL.encode(words)
                
                if len(word_vectors) != len(words):
                    continue
                
                # Calculate pairwise cosine similarity
                similarity_matrix = cosine_similarity(word_vectors)
                
                # Extract upper triangle (excluding diagonal) to avoid duplicates
                n_words = len(words)
                similarities = []
                for i in range(n_words):
                    for j in range(i + 1, n_words):
                        similarities.append(similarity_matrix[i, j])
                
                if similarities:
                    # Topic coherence: average semantic similarity within topic
                    avg_coherence = np.mean(similarities)
                    coherence_scores.append(max(avg_coherence, eps))
                    
            except Exception:
                # Skip this topic if embedding fails
                continue
        
        if coherence_scores:
            return np.clip(np.mean(coherence_scores), eps, 1.0)
        else:
            return eps

    except Exception:
        return eps


def evaluate_cluster_count(n_clusters: int, n_docs: int, eps: float = EPSILON) -> float:
    """
    Evaluate the appropriateness of cluster count with emphasis on practical utility.
    
    This function prioritizes cluster quality over quantity, considering:
    - Interpretability: Fewer, well-defined clusters are more valuable
    - Coverage: Each cluster should represent meaningful document subsets
    - Balance: Avoid extreme clustering (too few/many clusters)
    - Domain knowledge: Academic papers benefit from distinct thematic clusters
    
    Args:
        n_clusters: Number of clusters found by the model
        n_docs: Total number of documents in the dataset
        eps: Minimum epsilon value returned for invalid scenarios
        
    Returns:
        Validity score between 0 and 1, where 1 is optimal
    """
    try:
        # Zero clusters is invalid
        if n_clusters <= 0:
            return eps
            
        # Single cluster provides limited insight 
        if n_clusters == 1:
            return eps * 0.3
            
        # Absolute limits based on practicality using constants
        min_practical_clusters = max(MIN_CLUSTERS, n_docs // MAX_CLUSTER_RATIO)
        max_practical_clusters = min(n_docs // MIN_CLUSTER_RATIO, 100)
        
        if n_clusters < min_practical_clusters or n_clusters > max_practical_clusters:
            return eps * 0.1
        
        # Define quality zones with different scoring philosophies
        if n_clusters <= 5:
            # Very focused clusters - excellent for interpretability
            # Score based on how well they can cover the dataset
            coverage_ratio = n_clusters / np.min([n_docs // 200, 10])
            return np.clip(coverage_ratio, eps, 1.0)
            
        elif n_clusters <= 15:
            # Optimal range for academic topic modeling
            # Each cluster represents ~260-1940 documents (good granularity)
            baseline_score = 0.9
            # Slight preference for middle of this range (8-12 clusters)
            if 8 <= n_clusters <= 12:
                bonus = 0.05
            else:
                bonus = 0.02
            return np.clip(baseline_score + bonus, eps, 1.0)
            
        elif n_clusters <= 30:
            # Detailed clustering - good for comprehensive analysis
            # Score decreases gradually (more clusters = lower interpretability)
            distance_from_optimal = abs(n_clusters - 20) / 20
            return np.clip(0.7 - distance_from_optimal * 0.4, eps, 1.0)
            
        elif n_clusters <= 50:
            # Fine-grained clustering - limited practical value
            return eps + 0.2
            
        else:
            # Too many clusters - likely overfitting
            # More stringent penalty for excessive clusters
            excess_ratio = n_clusters / 50
            penalty = eps * (0.1 ** min(excess_ratio, 3))  # Exponential penalty
            return penalty
            
    except Exception:
        return eps

def _get_umap_embeddings(model, labels):
    """
    Helper function to extract UMAP embeddings from the model safely.
    
    Args:
        model: Trained BERTopic model
        labels: HDBSCAN labels array
        
    Returns:
        UMAP embeddings array or None if extraction fails
    """
    mask = labels != -1  # Filter out noise
    
    umap_embeddings = None
    
    # Try different methods to get UMAP embeddings with priority order
    if hasattr(model, 'umap_embeddings_') and model.umap_embeddings_ is not None:
        umap_embeddings = model.umap_embeddings_[mask]
    elif hasattr(model, 'umap_model') and hasattr(model.umap_model, 'embedding_'):
        umap_embeddings = model.umap_model.embedding_[mask]
    elif hasattr(model, 'umap_model') and hasattr(model.umap_model, '_raw_data'):
        umap_embeddings = model.umap_model._raw_data[mask]
    
    return umap_embeddings if (umap_embeddings is not None and 
                              len(umap_embeddings) > 0 and 
                              umap_embeddings.shape[1] >= 2) else None


def _compute_silhouette_score_normalized(embeddings, labels):
    """
    Compute normalized silhouette score.
    
    Args:
        embeddings: UMAP embeddings
        labels: Cluster labels
        
    Returns:
        Silhouette score normalized to [0, 1] range
    """
    try:
        s_score = silhouette_score(embeddings, labels)
        return (s_score + 1) / 2  # Convert from [-1,1] to [0,1]
    except Exception:
        return 0.5  # Default value on error


def _compute_ch_score_normalized(embeddings, labels):
    """
    Compute normalized Calinski-Harabasz score.
    
    Args:
        embeddings: UMAP embeddings
        labels: Cluster labels
        
    Returns:
        CH score normalized to [0, 1] range
    """
    try:
        ch_score = calinski_harabasz_score(embeddings, labels)
        
        if ch_score > 0:
            n_clusters = len(set(labels))
            n_samples = len(embeddings)
            
            # CH score normalization: more realistic approach
            # CH scores typically range within (log n, log n²)
            max_expected_ch = n_samples * np.log(n_clusters + 1) * 10  # Empirical value
            ch_score_scaled = min(ch_score / max_expected_ch, 1.0)
            # More conservative lower bound
            return max(ch_score_scaled, 0.01)
        else:
            return 0.0
            
    except Exception:
        return 0.5  # Default value on error


def compute_cluster_score(model, eps=EPSILON):
    """
    Compute clustering quality using Silhouette and Calinski-Harabasz scores.
    
    Args:
        model: Trained BERTopic model
        eps: Minimum epsilon value
        
    Returns:
        Combined cluster quality score between 0 and 1
    """
    try:
        # Get HDBSCAN labels
        if not hasattr(model, 'hdbscan_model') or model.hdbscan_model is None:
            return eps

        labels = model.hdbscan_model.labels_
        if labels is None or len(labels) == 0:
            return eps

        mask = labels != -1  # Filter out noise
        
        # Check minimum cluster requirement
        unique_labels = set(labels[mask])
        if len(unique_labels) < MIN_CLUSTERS:
            return eps

        # Get UMAP embeddings using helper function
        umap_embeddings = _get_umap_embeddings(model, labels[mask])
        if umap_embeddings is None:
            return eps

        # Compute normalized scores using helper functions
        s_score_scaled = _compute_silhouette_score_normalized(umap_embeddings, labels[mask])
        ch_score_scaled = _compute_ch_score_normalized(umap_embeddings, labels[mask])

        # Combined weighted score (Silhouette weighted higher)
        combined_score = 0.6 * s_score_scaled + 0.4 * ch_score_scaled
        return np.clip(combined_score, eps, 1.0)

    except Exception:
        return eps

def get_adaptive_sampler(study_name: str, dataset_size: int) -> TPESampler:
    """
    Create an enhanced TPE sampler following clustering optimization best practices.
    
    Key improvements:
    - Enhanced parameter relationships consideration
    - Proper Bayesian priors for better exploration
    - Optimized for mixed continuous/categorical parameter spaces
    """
    
    # Enhanced TPE Sampler: Compatible with dynamic value spaces
    return TPESampler(
        consider_prior=True,      # Use Bayesian mixture more explicitly
        prior_weight=1.0,         # Strong prior weight for categorical parameters
        consider_magic_clip=True, # Clips extremes adaptively
        consider_endpoints=False, # Exclude boundary values
        warn_independent_sampling=False,  # Suppress warning for dynamic search space
        seed=42
    )

def get_advanced_pruner(trial_count: int) -> MedianPruner:
    """
    Create adaptive pruner with best practices for clustering.
    
    Best Practices:
    1. Early pruning: Aggressive cost-saving for expensive evaluations
    2. Mid optimization: Balanced approach with moderate pruning
    3. Late optimization: Conservative pruning for final refinement
    """
    
    # Early pruning: Aggressive cost-saving
    if trial_count < EARLY_PRUNING_THRESHOLD:
        return HyperbandPruner(
            min_resource=1,
            max_resource=10,
            reduction_factor=3,
            bootstrap_count=0  # Compatible with fixed max_resource
        )
    
    # Mid optimization: Balanced approach  
    elif trial_count < MID_OPTIMIZATION_THRESHOLD:
        return SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=2
        )
    
    # Late optimization: Conservative pruning
    else:
        return MedianPruner(
            n_startup_trials=20,        # Wait for enough trials
            n_warmup_steps=5,           # Some steps without pruning
            interval_steps=5            # Consider pruning every 5 steps
        )

def _suggest_clustering_parameters(trial: optuna.Trial, dataset_size: int) -> tuple[int, int]:
    """
    Suggest clustering parameters with constraints.
    
    Args:
        trial: Optuna trial object
        dataset_size: Size of the dataset
        
    Returns:
        Tuple of (min_cluster_size, min_samples)
    """
    # Core clustering parameters (optimized for practical cluster counts)
    min_cluster_size_lower = max(5, dataset_size // MAX_CLUSTER_RATIO)
    min_cluster_size_upper = min(100, dataset_size // MIN_CLUSTER_RATIO)
    min_cluster_size = trial.suggest_int("min_cluster_size", min_cluster_size_lower, min_cluster_size_upper)
    
    # min_samples scales with min_cluster_size (important constraint!)
    min_samples_max = max(3, int(min_cluster_size * 0.8))
    min_samples = trial.suggest_int("min_samples", 3, min_samples_max)
    
    return min_cluster_size, min_samples


def _suggest_vectorization_parameters(trial: optuna.Trial, dataset_size: int) -> tuple[List[int], int, float]:
    """
    Suggest text vectorization parameters with constraints.
    
    Args:
        trial: Optuna trial object
        dataset_size: Size of the dataset
        
    Returns:
        Tuple of (ngram_range, min_df, max_df)
    """
    ngram_range = trial.suggest_categorical("ngram_range", [[1,2], [1,3]])
    
    # min_df vs max_df relationship (crucial!)
    min_df = trial.suggest_int("min_df", 2, min(10, dataset_size // 100))
    # max_df is a float in (0, 1], must be > min_df/doc_count
    min_df_ratio = min_df / dataset_size
    max_df_min = min_df_ratio + 0.01  # ensure max_df > min_df/doc_count
    max_df = trial.suggest_float("max_df", max_df_min, 0.95)
    
    return ngram_range, min_df, max_df


def _suggest_umap_parameters(trial: optuna.Trial, dataset_size: int) -> tuple[int, int, float, float]:
    """
    Suggest UMAP dimensionality reduction parameters.
    
    Args:
        trial: Optuna trial object  
        dataset_size: Size of the dataset
        
    Returns:
        Tuple of (n_neighbors, n_components, min_dist, spread)
    """
    # UMAP parameters with truly appropriate bounds
    practical_max = min(int(dataset_size * UMAP_NEIGHBORS_RATIO), UMAP_MAX_NEIGHBORS)
    
    # UMAP n_neighbors should be independent of clustering parameters
    n_neighbors = trial.suggest_int("n_neighbors", 5, practical_max)
    
    n_components = trial.suggest_int("n_components", 
                                  max(UMAP_MIN_COMPONENTS, int(np.log10(dataset_size))), 
                                  min(UMAP_MAX_COMPONENTS, dataset_size // 100))
    
    min_dist = trial.suggest_float("min_dist", 0.0, 0.5)
    spread = trial.suggest_float("spread", 0.8, 1.3)  # Narrower range for more stable results
    
    return n_neighbors, n_components, min_dist, spread


def suggest_constrained_parameters(trial: optuna.Trial, dataset_size: int) -> Hyperparameters:
    """
    Suggest parameters with intelligent constraints and interdependencies.
    
    Best Practice: Implement parameter constraints to avoid invalid combinations:
    - min_samples ≤ min_cluster_size (clustering constraint)
    - max_df > min_df (vectorization constraint)
    - n_neighbors ≤ dataset_size (computational constraint)
    - Focus on practical cluster count ranges (5-50 clusters preferred)
    """
    
    # Get parameter groups using helper functions
    min_cluster_size, min_samples = _suggest_clustering_parameters(trial, dataset_size)
    ngram_range, min_df, max_df = _suggest_vectorization_parameters(trial, dataset_size)
    n_neighbors, n_components, min_dist, spread = _suggest_umap_parameters(trial, dataset_size)
    
    return Hyperparameters(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

def _create_bertopic_model(params: Hyperparameters) -> BERTopic:
    """
    Create a BERTopic model with the given hyperparameters.
    
    Args:
        params: Hyperparameter configuration
        
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
        metric='cosine',
        min_dist=params.min_dist,
        spread=params.spread,
        random_state=42,
        low_memory=False  # Better for pruned early termination
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric='euclidean',
        prediction_data=False
    )
    
    return BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=EMBEDDING_MODEL,
        calculate_probabilities=False,
        verbose=False
    )


def _calculate_composite_score(coherence: float, cluster_score: float, cluster_validity: float,
                              eps: float = EPSILON) -> float:
    """
    Calculate weighted composite score from individual components.
    
    Args:
        coherence: Topic coherence score
        cluster_score: Clustering quality score
        cluster_validity: Cluster count validity score
        eps: Minimum epsilon value
        
    Returns:
        Composite optimization score between 0 and 1
    """
    # Check which scores are valid for weight adjustment
    valid_scores = {
        'coherence': coherence > eps,
        'cluster': cluster_score > eps,
        'validity': cluster_validity > eps
    }

    if sum(valid_scores.values()) == 0:
        return eps

    # Adjust weights for valid scores only and normalize
    adjusted_weights = {}
    total_weight = sum(SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS.keys() if valid_scores[k])
    
    for key in SCORE_WEIGHTS:
        if valid_scores[key]:
            adjusted_weights[key] = SCORE_WEIGHTS[key] / total_weight
        else:
            adjusted_weights[key] = 0

    # Calculate composite score
    base_score = (
        adjusted_weights['coherence'] * coherence +
        adjusted_weights['cluster'] * cluster_score +
        adjusted_weights['validity'] * cluster_validity
    )
    
    return np.clip(base_score, eps, 1.0)


def objective(trial: optuna.Trial, texts: List[str], text_embeddings: np.ndarray, eps: float = EPSILON) -> float:
    """
    Objective function for BERTopic hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        texts: List of input texts
        text_embeddings: Pre-computed text embeddings
        eps: Minimum epsilon value for score calculations
        
    Returns:
        Composite optimization score (0-1)
    """
    
    # Step 1: Suggest constrained parameters using best practices
    dataset_size = len(texts)
    params = suggest_constrained_parameters(trial, dataset_size)

    try:
        # Step 2: Train model using helper function
        model = _create_bertopic_model(params)

        # Step 3: Efficient evaluation with early termination hooks
        topics, _ = model.fit_transform(texts, embeddings=text_embeddings)
        
        # Quick validation checkpoint for pruning
        topic_info = model.get_topic_info()
        n_clusters = len(topic_info[topic_info['Topic'] != -1])
        
        # Early exit for obvious failures (helps pruning)
        if n_clusters < 2 or n_clusters > dataset_size // 5:
            return eps * 0.1  # Very low score for clear failures
        
        # Step 4: Multi-metric evaluation
        coherence = compute_coherence(model, eps=eps)
        cluster_score = compute_cluster_score(model, eps=eps)
        cluster_validity = evaluate_cluster_count(n_clusters, dataset_size, eps)

        # Step 5: Calculate composite score using helper function
        base_score = _calculate_composite_score(coherence, cluster_score, cluster_validity, eps)
        
        # Step 6: Simplified scoring without penalties
        final_score = base_score
        
        # Store key metrics for analysis
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("base_score", base_score)

        return np.clip(final_score, eps, 1.0)

    except Exception:
        # Graceful failure handling for clustering optimization
        return eps

def run_one_category(category: str, timeout: int = 10*60, storage: Optional[str] = None) -> optuna.Study:
    """
    Run hyperparameter optimization for a paper category.
    
    Args:
        category: Paper category to optimize
        timeout: Optimization timeout in seconds
        storage: Optuna storage backend
        
    Returns:
        Completed Optuna study
    """
    
    # Step 1: Load and prepare data efficiently
    papers = get_papers(category)
    text_embeddings = get_text_embeddings(category)
    texts = [EMBEDDING_MODEL.get_input_text(paper) for paper in papers]
    dataset_size = len(texts)
    del papers
    gc.collect()

    # Step 2: Create or load existing study with advanced configuration
    study_name = f"clustering_optimization_{category}_{dataset_size}"
    
    # Create new study with best practices
    study = optuna.create_study(
        storage=storage,
        load_if_exists=True,

        direction="maximize",
        study_name=study_name,
        sampler=get_adaptive_sampler(study_name, dataset_size),
        pruner=get_advanced_pruner(dataset_size)
    )

    # Step 3: Run optimization with best practices
    try:
        # For small datasets, use fewer trials but longer timeout per trial
        n_trials = max(50, min(200, dataset_size // 20))

        study.optimize(
            lambda trial: objective(trial, texts, text_embeddings),
            n_trials=n_trials,
            # timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=True,
            catch=(Exception,)  # Catch all exceptions gracefully
        )
        
        # Step 4: Post-optimization analysis and logging
        if len(study.trials) > 0:
            best_trial = study.best_trial
            print(f"\n=== Optimization Results ===")
            print(f"Best score: {best_trial.value:.4f}")
            print(f"Best parameters: {best_trial.params}")
            print(f"Total trials: {len(study.trials)}")
            print(f"Optimization completed successfully")

    except KeyboardInterrupt:
        print(f"\nOptimization interrupted. Completed {len(study.trials)} trials.")
    except Exception as e:
        print(f"Optimization error: {e}")
        # Still return study with partial results

    return study

if __name__ == "__main__":
    category = "physics.geo-ph"
    
    model_path = f"./models/{category}"
    os.makedirs(model_path, exist_ok=True)

    study_storage_path = f"sqlite:///{model_path}/search_params.db"
    study = run_one_category(category, timeout=20*60, storage=study_storage_path)

    params_storage_path = f"{model_path}/best_params.json"
    with open(params_storage_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
