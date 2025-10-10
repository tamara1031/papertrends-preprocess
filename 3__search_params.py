from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import os
import pickle
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
from typing import List, Dict

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model, CustomEmbeddingModel, get_category_codes
from common.memory_utils import (
    log_memory_usage, force_memory_cleanup, check_memory_threshold,
    get_dataset_memory_estimate, recommend_dataset_limit
)
from common.score_utils import compute_silhouette_score, compute_dbcv_score

# Suppress expected numerical warnings (validated as safe)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='hdbscan.validity')
warnings.filterwarnings('ignore', message='overflow encountered in power')
warnings.filterwarnings('ignore', message='divide by zero encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

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
        """Get number of trials based on dataset size."""
        if dataset_size <= 5000:
            return 20
        elif dataset_size <= 20000:
            return 40
        else:
            return 60
    
    @staticmethod
    def get_default_timeout(dataset_size: int) -> Optional[int]:
        """Get timeout in minutes based on dataset size."""
        if dataset_size <= 10000:
            return None
        elif dataset_size <= 50000:
            return 120
        else:
            return 240
    
    @staticmethod
    def get_adaptive_weights(dataset_size: int) -> Dict[str, float]:
        """Get balanced weights for both metrics."""
        return {
            'cluster_shape': 0.50,
            'clustering_quality': 0.50
        }
    
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
        return np.load(filepath, mmap_mode='r')
    except FileNotFoundError:
        raise FileNotFoundError(f"Text embeddings not found at {filepath}")

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

def create_bertopic_model(params: Hyperparameters, embedding_model: CustomEmbeddingModel) -> BERTopic:
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
        embedding_model=embedding_model,
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
        print(f"Warning: Failed to get basic model info: {e}")
        return {'n_topics': 0, 'top_cluster_sizes': []}

def compute_cluster_quality_score(
    model: BERTopic, 
    original_embeddings: np.ndarray,
    documents: List[str] = None,
    weights: Dict[str, float] = None
) -> float:
    """Compute combined clustering quality score with minimal memory usage."""
    try:
        dataset_size = len(original_embeddings)
        
        # Extract necessary data from model first
        labels = model.hdbscan_model.labels_
        umap_embedding = model.umap_model.embedding_
        topic_info = model.get_topic_info()
        
        # Get basic info first (lightweight operation)
        basic_info = _get_basic_model_info(topic_info)
        
        # Use provided weights or default weights
        if weights is None:
            weights = {
                'cluster_shape': 0.50,
                'clustering_quality': 0.50
            }
        
        # Initialize scores
        silhouette_umap_score = 0.0
        dbcv_basis_score = 0.0
        
        # Compute metrics only if needed (avoid unnecessary computation)
        if weights['cluster_shape'] > 0.00:
            silhouette_umap_score = compute_silhouette_score(labels, umap_embedding)
            # Clean up immediately after silhouette computation
            force_memory_cleanup()
        
        if weights['clustering_quality'] > 0.00:
            dbcv_basis_score = compute_dbcv_score(labels, original_embeddings)
            # Clean up immediately after DBCV computation
            force_memory_cleanup()
        
        # Calculate final score
        final_score = (
            weights['cluster_shape'] * silhouette_umap_score +
            weights['clustering_quality'] * dbcv_basis_score
        )
        
        # Output results (minimal string operations)
        print(f"Topics: {basic_info['n_topics']}, Top sizes: {basic_info['top_cluster_sizes']}")
        
        # Build output strings efficiently
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
        
        # Clean up all local variables
        del basic_info, weights, score_parts, weight_parts, labels, umap_embedding, topic_info
        
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
    """Create TPE sampler optimized for clustering."""
    startup_trials = max(10, min(20, int(n_trials * 0.15)))
    
    if dataset_size <= 10000:
        ei_candidates = max(16, min(32, int(n_trials * 0.20)))
    elif dataset_size <= 50000:
        ei_candidates = max(20, min(40, int(n_trials * 0.25)))
    else:
        ei_candidates = max(24, min(48, int(n_trials * 0.30)))
    
    return TPESampler(
        n_startup_trials=startup_trials,
        n_ei_candidates=ei_candidates,
        multivariate=True,
        group=False,
        prior_weight=1.0,
        warn_independent_sampling=True,
        seed=42
    )

def create_median_pruner(n_trials: int = 100, dataset_size: int = 10000) -> MedianPruner:
    """Create median pruner optimized for clustering."""
    startup_trials = max(10, min(20, int(n_trials * 0.15)))
    
    if dataset_size <= 10000:
        warmup_steps = 3
    elif dataset_size <= 50000:
        warmup_steps = 5
    else:
        warmup_steps = 7
    
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
    """Optuna objective function for hyperparameter optimization with aggressive memory management."""
    dataset_size = len(texts)
    model = None
    embeddings_copy = None
    
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
            
        # Fit model and get topics
        topics, _ = model.fit_transform(texts, embeddings=embeddings_copy)
        
        # Immediately clean up embeddings copy after fit_transform
        del embeddings_copy
        embeddings_copy = None
        force_memory_cleanup()
        
        # Evaluate clustering quality (this will use the model's internal data)
        weights = OptimizationConfig.get_adaptive_weights(dataset_size)
        score = compute_cluster_quality_score(model, text_embeddings, documents=texts, weights=weights)
        
        # Store evaluation metrics
        trial.set_user_attr("score", float(score))
        
        # Immediately clean up model after evaluation
        cleanup_bertopic_model(model)
        del model
        model = None
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
        # Aggressive cleanup in finally block
        if embeddings_copy is not None:
            del embeddings_copy
        if model is not None:
            cleanup_bertopic_model(model)
            del model
        force_memory_cleanup(aggressive=True)  


# ============================================================================
# Embedding Model Management
# ============================================================================

EMBEDDING_MODEL = get_custom_embedding_model()

def _memory_cleanup_callback(study, trial):
    """Memory cleanup callback with monitoring."""
    memory_info = log_memory_usage(f"Trial {trial.number}", verbose=False)
    
    # If memory usage is high, perform aggressive cleanup
    if memory_info['rss_mb'] > 1500:  # More than 1.5GB
        print(f"⚠️  High memory usage detected: {memory_info['rss_mb']:.1f} MB")
        force_memory_cleanup(aggressive=True)
        log_memory_usage(f"After aggressive cleanup (Trial {trial.number})", verbose=False)
    else:
        force_memory_cleanup()

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
    embedding_model = EMBEDDING_MODEL
    
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
                # Enhanced memory cleanup callback every 3 trials with memory monitoring
                lambda study, trial: (
                    _memory_cleanup_callback(study, trial) if trial.number % 3 == 0 and trial.number > 0 else None
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
        print("🧹 Performing final memory cleanup...")
        del texts, text_embeddings, embedding_model
        force_memory_cleanup(aggressive=True)
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
                raise
            except Exception as e:
                print(f"❌ Failed category {category}: {e}")
                continue
                
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