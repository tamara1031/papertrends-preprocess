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
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
# Note: gensim/topic coherence metrics intentionally excluded due to numpy version compatibility issues

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model

# Constants
EPSILON = 1e-6
EMBEDDING_MODEL = get_custom_embedding_model()

# Clustering constraints
MIN_CLUSTERS = 2
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


def _get_umap_embeddings(model, all_labels):
    """
    Helper function to extract UMAP embeddings from the model safely.
    
    Args:
        model: Trained BERTopic model
        all_labels: Full HDBSCAN labels array (including noise points as -1)
        
    Returns:
        Tuple of (umap_embeddings, filtered_labels) or (None, None) if extraction fails
    """
    mask = all_labels != -1  # Filter out noise
    print(f"DEBUG: UMAP extraction - mask sum: {np.sum(mask)}/{len(mask)}")
    
    umap_embeddings = None
    
    # Try different methods to get UMAP embeddings with priority order
    print(f"DEBUG: Model has umap_embeddings_: {hasattr(model, 'umap_embeddings_')}")
    if hasattr(model, 'umap_embeddings_'):
        print(f"DEBUG: model.umap_embeddings_ is None: {model.umap_embeddings_ is None}")
        
    if hasattr(model, 'umap_model'):
        print(f"DEBUG: UMAP model type: {type(model.umap_model)}")
        print(f"DEBUG: UMAP model has embedding_: {hasattr(model.umap_model, 'embedding_')}")
        print(f"DEBUG: UMAP model has _raw_data: {hasattr(model.umap_model, '_raw_data')}")
        
        if hasattr(model.umap_model, 'embedding_'):
            emb_shape = model.umap_model.embedding_.shape if model.umap_model.embedding_ is not None else 'None'
            print(f"DEBUG: UMAP embedding_ shape: {emb_shape}")
    
    # Extract embeddings for all points (including noise)
    if hasattr(model, 'umap_embeddings_') and model.umap_embeddings_ is not None:
        print("DEBUG: Using model.umap_embeddings_")
        umap_embeddings = model.umap_embeddings_
    elif hasattr(model, 'umap_model') and hasattr(model.umap_model, 'embedding_'):
        print("DEBUG: Using model.umap_model.embedding_")
        umap_embeddings = model.umap_model.embedding_
    elif hasattr(model, 'umap_model') and hasattr(model.umap_model, '_raw_data'):
        print("DEBUG: Using model.umap_model._raw_data")
        umap_embeddings = model.umap_model._raw_data
    
    if umap_embeddings is not None:
        print(f"DEBUG: Retrieved embeddings shape: {umap_embeddings.shape}")
        
        # Now apply the mask to both embeddings and labels
        masked_embeddings = umap_embeddings[mask]
        masked_labels = all_labels[mask]
        
        valid = (masked_embeddings is not None and 
                len(masked_embeddings) > 0 and 
                masked_embeddings.shape[1] >= 2)
        print(f"DEBUG: Masked embeddings valid: {valid}")
        print(f"DEBUG: Masked embeddings shape: {masked_embeddings.shape}")
        print(f"DEBUG: Masked labels shape: {masked_labels.shape}")
        
        return (masked_embeddings, masked_labels) if valid else (None, None)
    else:
        print("DEBUG: No embeddings retrieved")
        return (None, None)


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


def _compute_davies_bouldin_score_normalized(embeddings, labels):
    """
    Compute normalized Davies-Bouldin score.
    
    Davies-Bouldin score normalization based on typical clustering quality ranges:
    - Excellent (DB < 0.4): mapped to [0.8, 1.0]
    - Good (0.4 ≤ DB < 0.8): mapped to [0.6, 0.8] 
    - Moderate (0.8 ≤ DB < 1.2): mapped to [0.3, 0.6]
    - Poor (1.2 ≤ DB < 2.0): mapped to [0.1, 0.3]
    - Very Poor (DB ≥ 2.0): mapped to [0.01, 0.1]
    
    Note: Lower DB scores indicate better clustering (inverse relationship)
    
    Args:
        embeddings: UMAP embeddings
        labels: Cluster labels
        
    Returns:
        DB score normalized to [0, 1] range where higher values indicate better clustering
    """
    try:
        db_score = davies_bouldin_score(embeddings, labels)
        
        # DB score is inversely related to clustering quality (lower = better)
        # Reverse the relationship for consistent [0,1] scale with other metrics
        
        if db_score <= 0.4:
            # Excellent clustering - map (0, 0.4] to [0.8, 1.0]
            normalized = 0.8 + (0.4 - db_score) / 0.4 * 0.2
        elif db_score <= 0.8:
            # Good clustering - map (0.4, 0.8] to [0.6, 0.8]
            normalized = 0.6 + (0.8 - db_score) / 0.4 * 0.2
        elif db_score <= 1.2:
            # Moderate clustering - map (0.8, 1.2] to [0.3, 0.6]
            normalized = 0.3 + (1.2 - db_score) / 0.4 * 0.3
        elif db_score <= 2.0:
            # Poor clustering - map (1.2, 2.0] to [0.1, 0.3]
            normalized = 0.1 + (2.0 - db_score) / 0.8 * 0.2
        else:
            # Very poor clustering - map (2.0, ∞) to [0.01, 0.1]
            normalized = 0.01 + min(1.0 / db_score, 0.09)
        
        return np.clip(normalized, 0.01, 1.0)
        
    except Exception:
        return 0.5  # Default value on error




def _compute_ch_score_normalized(embeddings, labels):
    """
    Compute normalized Calinski-Harabasz score using empirically-based ranges.
    
    Updated CH score normalization based on actual clustering performance ranges:
    - Excellent (CH≥100,000): mapped to [0.9, 1.0]
    - Very Good (10,000≤CH<100,000): mapped to [0.8, 0.9] 
    - Good (1,000≤CH<10,000): mapped to [0.6, 0.8]
    - Moderate (100≤CH<1,000): mapped to [0.3, 0.6]
    - Poor (0<CH<100): mapped to [0.01, 0.3]
    
    Args:
        embeddings: UMAP embeddings
        labels: Cluster labels
        
    Returns:
        CH score normalized to [0, 1] range where higher values indicate better clustering
    """
    try:
        ch_score = calinski_harabasz_score(embeddings, labels)
        
        if ch_score > 0:
            # Updated normalization based on realistic CH score ranges observed in practice:
            # - Excellent clustering: CH ≥ 100,000 (very tight clusters)
            # - Very Good clustering: 10,000 ≤ CH < 100,000 
            # - Good clustering: 1,000 ≤ CH < 10,000
            # - Moderate clustering: 100 ≤ CH < 1,000
            # - Poor clustering: CH < 100
            
            if ch_score >= 100000:
                # Excellent clustering - map [100,000, ∞) to [0.9, 1.0]
                # Use logarithmic scaling to handle very large values gracefully
                normalized = 0.9 + min(np.log10(ch_score / 100000) / 10, 0.1)
            elif ch_score >= 10000:
                # Very Good clustering - map [10,000, 100,000) to [0.8, 0.9]
                normalized = 0.8 + (ch_score - 10000) / 90000 * 0.1
            elif ch_score >= 1000:
                # Good clustering - map [1,000, 10,000) to [0.6, 0.8]
                normalized = 0.6 + (ch_score - 1000) / 9000 * 0.2
            elif ch_score >= 100:
                # Moderate clustering - map [100, 1,000) to [0.3, 0.6]
                normalized = 0.3 + (ch_score - 100) / 900 * 0.3
            else:
                # Poor clustering - map (0, 100) to [0.01, 0.3]
                normalized = 0.01 + (ch_score / 100) * 0.29
            
            return np.clip(normalized, 0.01, 1.0)
        else:
            return 0.01
            
    except Exception:
        return 0.5  # Default value on error

def compute_cluster_score(model, eps=EPSILON):
    """
    Compute clustering quality using Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    
    Enhanced evaluation combining three complementary clustering metrics:
    - Silhouette: Separation vs cohesion balance
    - Calinski-Harabasz: Inter-cluster vs intra-cluster variance ratio  
    - Davies-Bouldin: Intra-cluster density evaluation
    
    Note: Topic Coherence metrics (gensim) are excluded due to numpy version 
    compatibility issues in the current environment.
    
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
        umap_embeddings, filtered_labels = _get_umap_embeddings(model, labels)
        if umap_embeddings is None:
            return eps

        # Compute normalized scores using helper functions
        s_score_scaled = _compute_silhouette_score_normalized(umap_embeddings, filtered_labels)
        ch_score_scaled = _compute_ch_score_normalized(umap_embeddings, filtered_labels)
        db_score_scaled = _compute_davies_bouldin_score_normalized(umap_embeddings, filtered_labels)

        # Optimized weighting based on empirical analysis and numpy compatibility considerations:
        # Silhouette: 40% (separation-cohesion balance, most stable metric)
        # Calinski-Harabasz: 35% (inter-cluster separation, robust for academic papers) 
        # Davies-Bouldin: 25% (intra-cluster cohesion, complements other metrics)
        # 
        # Note: This 3-metric combination provides comprehensive evaluation without
        # dependency on gensim/topic coherence metrics that have numpy version conflicts
        combined_score = (0.40 * s_score_scaled + 0.35 * ch_score_scaled + 0.25 * db_score_scaled)
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
    - More exploration to break away from plateaus
    """
    
    # Enhanced TPE Sampler: Compatible with dynamic value spaces
    return TPESampler(
        consider_prior=True,      # Use Bayesian mixture more explicitly
        prior_weight=0.8,         # Reduced prior weight to encourage more exploration
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
        
        # Step 4: Cluster quality evaluation only
        cluster_score = compute_cluster_score(model, eps=eps)
        
        # Step 5: Use cluster_score as final_score
        final_score = cluster_score
        
        # Store key metrics for analysis
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("cluster_score", float(cluster_score))
        
        # Debug: Print cluster scores when they're high with individual metrics
        if final_score > 0.7:
            # Extract individual scores for debugging (recompute to get detailed metrics)
            debug_labels = model.hdbscan_model.labels_
            debug_umap_embeddings, debug_filtered_labels = _get_umap_embeddings(model, debug_labels)
            if debug_umap_embeddings is not None:
                s_score = _compute_silhouette_score_normalized(debug_umap_embeddings, debug_filtered_labels)
                ch_score = _compute_ch_score_normalized(debug_umap_embeddings, debug_filtered_labels)
                db_score = _compute_davies_bouldin_score_normalized(debug_umap_embeddings, debug_filtered_labels)
                print(f"High cluster score - Trial {trial.number}: "
                      f"s_score={s_score:.4f}, ch_score={ch_score:.4f}, db_score={db_score:.4f}, "
                      f"combined={cluster_score:.4f}, n_clusters={n_clusters}")

        return np.clip(final_score, eps, 1.0)

    except Exception as e:
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
