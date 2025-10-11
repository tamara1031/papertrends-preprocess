import os
import pickle
from pathlib import Path

from datetime import date
from typing import List, Optional, Tuple, Dict
import json
import numpy as np
import warnings

import torch
import optuna
from sklearn.feature_extraction.text import CountVectorizer
import optuna.exceptions
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from utils.custom_embedder import Specter2Embedder
from utils.memory_utils import force_memory_cleanup
from utils.score_utils import compute_silhouette_score, compute_dbcv_score
from utils.hyperparameter import Hyperparameters
from papertrends_dataset_lib.utils import ConfigLoader

# Suppress expected numerical warnings (validated as safe)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='hdbscan.validity')
warnings.filterwarnings('ignore', message='overflow encountered in power')
warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

# ============================================================================
# Singleton
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = Specter2Embedder(device=device)
CONFIG_LOADER = ConfigLoader(Path(__file__).parent / "config")

# ============================================================================
# Dataset Loader
# ============================================================================

def load_texts(category: str, subcategory: str) -> List[str]:
    """
    Load preprocessed paper titles for a given arXiv category/subcategory
    from the ./dataset directory.

    Args:
        category (str): Top-level arXiv category.
        subcategory (str): Subcategory.

    Returns:
        List[str]: List of paper titles. If not found, returns an empty list.
    """
    base_dir = Path("./dataset") / category / subcategory
    titles_path = base_dir / "titles.pkl"
    abstracts_path = base_dir / "abstracts.pkl"
    
    if not titles_path.exists() or not abstracts_path.exists():
        print(f"Dataset files not found for {category}/{subcategory}")
        return []
    
    with open(titles_path, "rb") as f:
        titles = pickle.load(f)
    with open(abstracts_path, "rb") as f:
        abstracts = pickle.load(f)
    
    return [EMBEDDING_MODEL.get_input_text(title, abstract) for title, abstract in zip(titles, abstracts)]

def load_text_embeddings(category: str, subcategory: str) -> np.ndarray:
    """Load text embeddings for a category/subcategory."""
    base_dir = Path("./dataset") / category / subcategory
    embeddings_path = base_dir / "embeddings.pkl"
    
    if not embeddings_path.exists():
        print(f"Embeddings file not found for {category}/{subcategory}")
        return np.array([])
    
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

# ============================================================================
# Memory Management
# ============================================================================

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
    
    # Constants
    UMAP_METRICS = ["cosine"]
    HDBSCAN_METRICS = ["euclidean", "manhattan"]
    TOP_N_WORDS_RANGE = (10, 20)
    NGRAM_RANGES = [[1, 3]]
    MIN_SAMPLES_MULTIPLIER_RANGE = (0.5, 1.0)
    
    @staticmethod
    def get_default_n_trials(dataset_size: int) -> int:
        """Get number of trials based on dataset size with increased trials for diversity."""
        if dataset_size <= 5000:
            return 50  # 30 → 50に増加
        elif dataset_size <= 20000:
            return 80  # 50 → 80に増加
        else:
            return 120  # 80 → 120に増加
    
    @staticmethod
    def get_default_timeout(dataset_size: int) -> Optional[int]:
        """Get timeout in minutes based on dataset size."""
        if dataset_size <= 10000:
            return 60  # タイムアウトを設定
        elif dataset_size <= 50000:
            return 90  # 短縮
        else:
            return 180  # 短縮
    
    @staticmethod
    def get_adaptive_weights(dataset_size: int) -> Dict[str, float]:
        """Get balanced weights for both metrics (scores in [-1, 1] range)."""
        return {
            'cluster_shape': 0.5,
            'clustering_quality': 0.5
        }
    
    @staticmethod
    def get_min_df_range(dataset_size: int) -> Tuple[int, int]:
        """Get min_df range based on dataset size."""
        if dataset_size <= 20000:
            min_val = 2
        elif dataset_size <= 50000:
            min_val = 3
        elif dataset_size <= 100000:
            min_val = 4
        else:
            min_val = 5
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
# Parameter Suggestion Functions
# ============================================================================

def suggest_optimal_hyperparameters(trial: optuna.Trial, dataset_size: int) -> Hyperparameters:
    """Suggest complete hyperparameter set."""
    # Clustering parameters
    min_cluster_size_range = OptimizationConfig.get_min_cluster_size_range(dataset_size)
    min_cluster_size = trial.suggest_int("min_cluster_size", *min_cluster_size_range)
    
    min_samples_multiplier = trial.suggest_float(
        "min_samples_multiplier",
        *OptimizationConfig.MIN_SAMPLES_MULTIPLIER_RANGE
    )
    min_samples = max(1, min(int(min_cluster_size * min_samples_multiplier), min_cluster_size))
    hdbscan_metric = trial.suggest_categorical("hdbscan_metric", OptimizationConfig.HDBSCAN_METRICS)
    
    # Vectorization parameters
    top_n_words = trial.suggest_int("top_n_words", *OptimizationConfig.TOP_N_WORDS_RANGE)
    ngram_range = trial.suggest_categorical("ngram_range", OptimizationConfig.NGRAM_RANGES)
    
    min_df_range = OptimizationConfig.get_min_df_range(dataset_size)
    min_df = trial.suggest_int("min_df", *min_df_range)
    
    max_df_range = OptimizationConfig.get_max_df_range(dataset_size)
    max_df = trial.suggest_int("max_df", *max_df_range)
    
    # UMAP parameters
    n_neighbors_range = OptimizationConfig.get_n_neighbors_range(dataset_size)
    n_neighbors = trial.suggest_int("n_neighbors", *n_neighbors_range)
    
    n_components_range = OptimizationConfig.get_n_components_range(dataset_size)
    n_components = trial.suggest_int("n_components", *n_components_range)
    
    umap_metric = trial.suggest_categorical("umap_metric", OptimizationConfig.UMAP_METRICS)
    
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

def create_bertopic_model(params: Hyperparameters) -> BERTopic:
    """Create BERTopic model with optimized parameter configuration."""
    vectorizer_model = CountVectorizer(
        stop_words="english",
        analyzer="word",
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
        low_memory=True
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric=params.hdbscan_metric,
        prediction_data=False
    )
    
    return BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=EMBEDDING_MODEL,
        top_n_words=params.top_n_words,
        calculate_probabilities=False,
        verbose=False
    )


# ============================================================================
# Evaluation Metrics
# ============================================================================

def _get_basic_model_info(topic_info: np.ndarray) -> dict:
    """Get basic information about the trained model with minimal memory usage."""
    try:
        # Filter out noise topics (-1) efficiently
        valid_topics_mask = topic_info['Topic'] != -1
        valid_topics = topic_info[valid_topics_mask]
        n_topics = len(valid_topics)
        
        if n_topics > 0:
            # Get only the count values (small array)
            cluster_sizes = valid_topics['Count'].values
            # Sort and get top 3 without creating large intermediate arrays
            top_sizes = np.sort(cluster_sizes)[::-1][:3]  # Top 3 largest clusters
        else:
            top_sizes = []
        
        # Clean up intermediate variables
        del valid_topics, cluster_sizes
        
        return {'n_topics': n_topics, 'top_cluster_sizes': top_sizes}
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt to be caught by outer try-except
        raise    
    except Exception as e:
        print(f"Failed to get basic model info: {e}")
        return {'n_topics': 0, 'top_cluster_sizes': []}

def compute_cluster_quality_score(
    labels: np.ndarray,
    embedding: np.ndarray,
    weights: Dict[str, float]
) -> float:
    """Compute combined clustering quality score with minimal memory usage."""
    try:
        # Compute metrics
        silhouette_score = compute_silhouette_score(labels, embedding)
        force_memory_cleanup()
        
        dbcv_score = compute_dbcv_score(labels, embedding)
        force_memory_cleanup()
        
        # Calculate final score
        final_score = (
            weights['cluster_shape'] * silhouette_score +
            weights['clustering_quality'] * dbcv_score
        )
        
        # Output results
        print(f"Silhouette: {silhouette_score:.4f}, DBCV: {dbcv_score:.4f}, Final: {final_score:.4f}")
        
        return final_score
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt to be caught by outer try-except
        raise
    except Exception as e:
        print(f"Error in compute_cluster_quality_score: {e}")
        return -1.0  # Return worst score on error (range: [-1, 1])  


# ============================================================================
# Optuna Configuration
# ============================================================================

def create_tpe_sampler(n_trials: int = 100, dataset_size: int = 10000) -> TPESampler:
    """Create TPE sampler optimized for clustering with increased random walk."""
    # ランダムウォークを大幅に増やして多様性を確保
    startup_trials = max(15, min(30, int(n_trials * 0.25)))  # 10% → 25%に大幅増加
    
    # ei_candidatesを適度に調整（多様性と効率のバランス）
    if dataset_size <= 10000:
        ei_candidates = max(24, min(40, int(n_trials * 0.25)))
    elif dataset_size <= 50000:
        ei_candidates = max(24, min(50, int(n_trials * 0.30)))
    else:
        ei_candidates = max(24, min(60, int(n_trials * 0.35)))
    
    return TPESampler(
        n_startup_trials=startup_trials,
        n_ei_candidates=ei_candidates,
        multivariate=True,
        group=False,
        prior_weight=1.0,  # より強い事前分布の重み
        warn_independent_sampling=True,
        seed=42
    )

def create_median_pruner(n_trials: int = 100, dataset_size: int = 10000) -> MedianPruner:
    """Create median pruner with delayed pruning for more exploration."""
    # startup_trialsを増やしてプルーニング開始を遅らせる
    startup_trials = max(20, min(40, int(n_trials * 0.30)))
    
    # warmup_stepsを適度に増やしてプルーニング判定を遅らせる
    if dataset_size <= 10000:
        warmup_steps = 6  # 10 → 6に調整
    elif dataset_size <= 50000:
        warmup_steps = 8  # 15 → 8に調整
    else:
        warmup_steps = 10  # 20 → 10に調整
    
    # interval_stepsを増やしてプルーニング判定頻度を下げる
    interval_steps = 3  # 1 → 3に増加
    
    return MedianPruner(
        n_startup_trials=startup_trials,
        n_warmup_steps=warmup_steps,
        interval_steps=interval_steps
    )


def objective_function(
    trial: optuna.Trial, 
    texts: List[str], 
    text_embeddings: np.ndarray
) -> float:
    """Optuna objective function for hyperparameter optimization."""
    model = None
    try:
        # Suggest hyperparameters and create model
        params = suggest_optimal_hyperparameters(trial, len(texts))
        model = create_bertopic_model(params)
        
        # Fit model
        topics, _ = model.fit_transform(texts, embeddings=text_embeddings)
        
        # Evaluate clustering quality
        weights = OptimizationConfig.get_adaptive_weights(len(texts))
        
        # Extract necessary data from model
        labels = model.hdbscan_model.labels_
        umap_embedding = model.umap_model.embedding_
        topic_info = model.get_topic_info()
        
        # Clean up model immediately after extracting data
        cleanup_bertopic_model(model)
        del model
        model = None
        force_memory_cleanup()
        
        # Get basic info and output
        basic_info = _get_basic_model_info(topic_info)
        print(f"Topics: {basic_info['n_topics']}, Top sizes: {basic_info['top_cluster_sizes']}")
        
        score = compute_cluster_quality_score(labels, umap_embedding, weights)
        
        # Store evaluation metrics
        trial.set_user_attr("score", float(score))
        
        return score
    
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt in trial {trial.number}")
        raise
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        trial.set_user_attr("error", str(e))
        return -1.0  # Return worst score on error (range: [-1, 1])
    finally:
        # Critical cleanup only (model may already be cleaned up)
        if model is not None:
            cleanup_bertopic_model(model)
            del model
        force_memory_cleanup()

# ============================================================================
# Main Optimization Pipeline
# ============================================================================

def _memory_cleanup_callback(study, trial):
    """Simple memory cleanup callback."""
    # Only cleanup if trial number is divisible by 10 (every 10 trials)
    if trial.number % 10 == 0 and trial.number > 0:
        force_memory_cleanup()

def optimize_category_clustering(
    category: str, 
    subcategory: str,
    timeout: Optional[int] = None, 
    n_trials: Optional[int] = None,
    storage: Optional[str] = None
) -> optuna.Study:
    """Run hyperparameter optimization for a specific arXiv category."""
    
    # Load and prepare data
    print(f"Loading data for {category}/{subcategory}")
    texts = load_texts(category, subcategory)
    text_embeddings = load_text_embeddings(category, subcategory)
    
    dataset_size = len(texts)
    print(f"Dataset size: {dataset_size:,} documents")
    
    # Adaptive configuration
    if n_trials is None:
        n_trials = OptimizationConfig.get_default_n_trials(dataset_size)
    if timeout is None:
        timeout_minutes = OptimizationConfig.get_default_timeout(dataset_size)
        timeout = timeout_minutes * 60 if timeout_minutes else None
    
    print(f"Settings: {n_trials} trials, {timeout//60 if timeout else 'None'} min timeout")
    
    # Create study
    study_name = f"clustering_optimization_{category}_{subcategory}_{dataset_size}"
    study = optuna.create_study(
        storage=storage,
        load_if_exists=True,  
        direction="maximize",
        study_name=study_name,
        sampler=create_tpe_sampler(n_trials, dataset_size),
        pruner=create_median_pruner(n_trials, dataset_size)
    )
    
    # Run optimization
    print(f"Starting optimization...")
    
    try:
        study.optimize(
            lambda trial: objective_function(trial, texts, text_embeddings),
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=True,
            catch=(ValueError, RuntimeError, MemoryError),
            callbacks=[_memory_cleanup_callback] if n_trials > 10 else []
        )
        
        # Display results
        _display_optimization_results(study)
            
    except KeyboardInterrupt:
        print(f"Optimization interrupted. Completed {len(study.trials)} trials.")
        raise
    except Exception as e:
        print(f"Optimization error: {e}")
        raise
    finally:
        # Critical cleanup only
        force_memory_cleanup()
    
    return study


def _display_optimization_results(study: optuna.Study) -> None:
    """Display optimization results."""
    if len(study.trials) == 0:
        print("No completed trials found.")
        return
    
    best_trial = study.best_trial
    pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    success_rate = (len(study.trials) - pruned_count) / len(study.trials)
    
    print(f"\nOptimization Results:")
    print(f"Best score: {best_trial.value:.4f}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Pruned trials: {pruned_count}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Best parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")


def save_optimization_results(study: optuna.Study, output_dir: str) -> None:
    """Save optimization results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best parameters
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"Results saved to: {best_params_path}")


# ============================================================================
# Main Execution
# ============================================================================

def process_one_category(category: str, subcategory: str):
    """Main execution function for hyperparameter optimization."""
    # Create output directory
    params_path = f"./params/{category}/{subcategory}"
    os.makedirs(params_path, exist_ok=True)
    
    # Run optimization
    study_storage_path = f"sqlite:///{params_path}/search_params.db"
    study = optimize_category_clustering(
        category=category,
        subcategory=subcategory,
        storage=study_storage_path
    )
    
    # Save results
    save_optimization_results(study, params_path)


if __name__ == "__main__":
    print("SPECTER2-BASED ACADEMIC PAPER CLUSTERING OPTIMIZATION")
    print("=" * 60)
    
    categories = CONFIG_LOADER.load_yaml("categories.yaml")
    
    print(f"Processing {len(categories)} arXiv categories:")
    for i, category in enumerate(categories, 1):
        print(f"  {i:2d}. {category}")
    
    print(f"\nStarting hyperparameter optimization...")
    print("-" * 60)
    
    total_subcategories = sum(len(category_items) for category_items in categories.values())
    processed_count = 0
    
    for category_name, category_items in categories.items():
        for subcategory in category_items:
            processed_count += 1
            print(f"\n[{processed_count}/{total_subcategories}] Processing {category_name}/{subcategory}")
            
            try:
                process_one_category(category_name, subcategory)
                print(f"Completed: {category_name}/{subcategory}")
            except KeyboardInterrupt:
                print(f"\nInterrupted by user. Processed {processed_count-1}/{total_subcategories} subcategories.")
                print(f"Results saved to: ./params/")
                import sys
                sys.exit(0)
            except Exception as e:
                print(f"Failed {category_name}/{subcategory}: {e}")
                continue
                
    print("\n" + "=" * 60)
    print("All subcategories processed successfully!")
    print("Results saved to: ./params/{category}/{subcategory}/")
    print("=" * 60)