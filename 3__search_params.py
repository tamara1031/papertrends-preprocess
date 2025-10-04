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
from sklearn.metrics import davies_bouldin_score
from sentence_transformers import util

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
    mask = all_labels != -1  # Filter out noise
    umap_embeddings = None
    
    # Extract embeddings for all points (including noise)
    if hasattr(model, 'umap_embeddings_') and model.umap_embeddings_ is not None:
        umap_embeddings = model.umap_embeddings_
    elif hasattr(model, 'umap_model') and hasattr(model.umap_model, 'embedding_'):
        umap_embeddings = model.umap_model.embedding_
    elif hasattr(model, 'umap_model') and hasattr(model.umap_model, '_raw_data'):
        umap_embeddings = model.umap_model._raw_data
    
    if umap_embeddings is not None:
        # Now apply the mask to both embeddings and labels
        masked_embeddings = umap_embeddings[mask]
        masked_labels = all_labels[mask]
        
        valid = (masked_embeddings is not None and 
                len(masked_embeddings) > 0 and 
                masked_embeddings.shape[1] >= 2)
        
        return (masked_embeddings, masked_labels) if valid else (None, None)
    else:
        return (None, None)


def _compute_davies_bouldin_score_normalized(embeddings, labels):
    try:
        return davies_bouldin_score(embeddings, labels)
        
    except Exception:
        return 2.0  # Davies-Bouldinでは低いほど良いので、中程度の悪いスコアを返す
        
def compute_cluster_score(model: BERTopic, eps=EPSILON):
    try:
        # Get HDBSCAN labels
        if not hasattr(model, 'hdbscan_model') or model.hdbscan_model is None:
            return -2.0

        labels = model.hdbscan_model.labels_
        if labels is None or len(labels) == 0:
            return -2.0

        mask = labels != -1  # Filter out noise
        
        # Check minimum cluster requirement
        unique_labels = set(labels[mask])
        if len(unique_labels) < MIN_CLUSTERS:
            return -2.0

        # Get UMAP embeddings using helper function
        umap_embeddings, filtered_labels = _get_umap_embeddings(model, labels)
        if umap_embeddings is None:
            return -2.0

        # Compute Davies-Bouldin score (lower is better, so return negative for maximization)
        db_score = _compute_davies_bouldin_score_normalized(umap_embeddings, filtered_labels)
        
        # Davies-Bouldinは低いほど良いので、負の値を返してOptunaに最大化させる
        return -db_score

    except Exception:
        return -2.0  # 例外時は悪いスコア（2.0）の負の値

def get_adaptive_sampler(study_name: str, dataset_size: int) -> TPESampler:
    
    # Enhanced TPE Sampler: Compatible with dynamic value spaces
    return TPESampler(
        consider_prior=True,      # Use Bayesian mixture more explicitly
        prior_weight=0.85,         # Reduced prior weight to encourage more exploration
        consider_magic_clip=True, # Clips extremes adaptively
        consider_endpoints=False, # Exclude boundary values
        warn_independent_sampling=False,  # Suppress warning for dynamic search space
        seed=42
    )

def get_advanced_pruner(trial_count: int) -> MedianPruner:
    
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
    # Core clustering parameters (optimized for practical cluster counts)
    min_cluster_size_lower = max(5, dataset_size // MAX_CLUSTER_RATIO)
    min_cluster_size_upper = min(100, dataset_size // MIN_CLUSTER_RATIO)
    min_cluster_size = trial.suggest_int("min_cluster_size", min_cluster_size_lower, min_cluster_size_upper)
    
    # min_samples scales with min_cluster_size (important constraint!)
    min_samples_max = max(3, int(min_cluster_size * 0.8))
    min_samples = trial.suggest_int("min_samples", 3, min_samples_max)
    
    return min_cluster_size, min_samples


def _suggest_vectorization_parameters(trial: optuna.Trial, dataset_size: int) -> tuple[List[int], int, float]:
    ngram_range = trial.suggest_categorical("ngram_range", [[1,2], [1,3]])
    
    # min_df vs max_df relationship (crucial!)
    min_df = trial.suggest_int("min_df", 2, min(10, dataset_size // 100))
    # max_df is a float in (0, 1], must be > min_df/doc_count
    min_df_ratio = min_df / dataset_size
    max_df_min = min_df_ratio + 0.01  # ensure max_df > min_df/doc_count
    max_df = trial.suggest_float("max_df", max_df_min, 0.95)
    
    return ngram_range, min_df, max_df


def _suggest_umap_parameters(trial: optuna.Trial, dataset_size: int) -> tuple[int, int, float, float]:
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
        
        # Step 4: Cluster quality evaluation only
        cluster_score = compute_cluster_score(model, eps=eps)
        
        # Store key metrics for analysis
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("cluster_score", float(cluster_score))
        
        return cluster_score

    except Exception as e:
        # Graceful failure handling for clustering optimization
        return eps

def run_one_category(category: str, timeout: int = 10*60, storage: Optional[str] = None) -> optuna.Study:
    
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
