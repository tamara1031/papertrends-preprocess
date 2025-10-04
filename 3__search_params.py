from typing import List, Optional, Union
from dataclasses import dataclass
import gc
import os
import pickle
import json
import numpy as np

import optuna
import optuna.exceptions
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model, CustomEmbeddingModel

from hdbscan.validity import validity_index
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Constants
EPSILON = 1e-6
EMBEDDING_MODEL = get_custom_embedding_model()

# Clustering constraints
MIN_CLUSTER_RATIO = 20  # Min cluster size = dataset_size // MIN_CLUSTER_RATIO
MAX_CLUSTER_RATIO = 500  # Max cluster size = dataset_size // MAX_CLUSTER_RATIO

# UMAP parameters
UMAP_NEIGHBORS_RATIO = 0.03  # Max n_neighbors = dataset_size * UMAP_NEIGHBORS_RATIO
UMAP_MAX_NEIGHBORS = 50  # Absolute maximum for n_neighbors
UMAP_MIN_COMPONENTS = 2
UMAP_MAX_COMPONENTS = 15


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

def _compute_words_coherence(words, embedding_model: CustomEmbeddingModel):
    """
    words_list: list of words (1トピックの単語リスト)
    returns: 平均cosine similarity
    """
    embeddings = embedding_model.embed(words)

    sims = []
    for w1, w2 in combinations(range(len(words)), 2):
        sim = cosine_similarity([embeddings[w1]], [embeddings[w2]])[0][0]
        sims.append(sim)
    
    return np.mean(sims)

def _compute_coherence_score(model: BERTopic, eps=EPSILON):
    """
    Calculate topic coherence score: document-count weighted average coherence of all topics.
    Returns score in range [0, 1].
    """
    embedding_model = model.embedding_model
    topic_words_dict = model.get_topics()
    topic_words_dict = {k: [w for w, _ in v] for k, v in topic_words_dict.items() if k != -1}

    if not topic_words_dict:
        return eps

    # Get topic document counts from model labels
    topic_counts = {}
    labels = model.hdbscan_model.labels_
    for label in labels:
        if label != -1:  # Exclude outliers
            topic_counts[label] = topic_counts.get(label, 0) + 1

    topic_coherences = []
    topic_weights = []
    
    for topic_id, words in topic_words_dict.items():
        if len(words) < 2:  # Skip topics with less than 2 words
            continue
        
        topic_coherence = _compute_words_coherence(words, embedding_model)
        topic_coherences.append(topic_coherence)
        
        # Use document count as weight
        doc_count = topic_counts.get(topic_id, 1)  # Default to 1 if not found
        topic_weights.append(doc_count)
    
    if not topic_coherences:
        return eps
    
    # Calculate document-count weighted average coherence
    topic_coherences = np.array(topic_coherences)
    topic_weights = np.array(topic_weights)
    
    weighted_avg_coherence = np.average(topic_coherences, weights=topic_weights)
    
    # Ensure the score is in [0, 1] range
    # cosine similarity can be negative, so we clip and normalize
    normalized_coherence = max(0.0, min(1.0, (weighted_avg_coherence + 1.0) / 2.0))
    
    return normalized_coherence

def _compute_dbcv_score(model: BERTopic, eps=EPSILON):
    try:
        embeddings = model.umap_model.embedding_
        labels = model.hdbscan_model.labels_
        distance_matrix = pairwise_distances(embeddings, metric='euclidean')
        
        # Compute raw DBCV score
        dbcv_score = validity_index(distance_matrix, labels)
        
        # DBCV score can be negative, so we normalize and clip to [0, 1]
        # Adding 1 to shift negative values to positive range, then normalize
        normalized_score = max(0.0, min(1.0, (dbcv_score + 1.0) / 2.0))
        
        return normalized_score
        
    except Exception as e:
        # Return fallback value on any error
        print(f"Warning: DBCV computation failed: {e}")
        return eps
        
def compute_cluster_score(model: BERTopic, eps=EPSILON):
    try:
        coherence_score = _compute_coherence_score(model, eps=eps)
        dbcv_score = _compute_dbcv_score(model, eps=eps)
        
        combined_score = 0.4 * coherence_score + 0.6 * dbcv_score
        
        return combined_score

    except Exception:
        return eps

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

def get_advanced_pruner() -> MedianPruner:
    """Simple but effective pruner for clustering optimization."""
    return MedianPruner(
        n_startup_trials=10,        # Wait for enough trials to start pruning
        n_warmup_steps=3,           # Some steps without pruning  
        interval_steps=3            # Consider pruning every 3 steps
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
        
        # Early termination if no valid clusters found
        if n_clusters == 0:
            trial.report(eps, 0)
            raise optuna.exceptions.TrialPruned()
        
        # Step 4: Evaluate clustering quality with combined coherence and DBCV scores
        combined_score = compute_cluster_score(model, eps=eps)
        
        # Store key metrics for analysis
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("combined_score", float(combined_score))
        
        return combined_score

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
        pruner=get_advanced_pruner()
    )

    # Step 3: Run optimization with appropriate trial count
    try:
        # Scale trial count based on dataset size
        n_trials = min(100, max(30, dataset_size // 50))

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
