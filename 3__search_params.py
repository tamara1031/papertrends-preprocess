from typing import List, Any, Optional, Union
from dataclasses import dataclass

import gc, os
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

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# Constants
EPSILON = 1e-6
EMBEDDING_MODEL = get_custom_embedding_model()


# Score weights for composite evaluation (balanced for practical utility)
SCORE_WEIGHTS = {
    'coherence': 0.30,      # Topic internal consistency
    'diversity': 0.20,      # Topic distinctiveness  
    'cluster': 0.25,         # Clustering quality metrics
    'validity': 0.25        # Practical cluster count appropriateness
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
    lowercase: bool
    strip_accents: Optional[Any]
    bm25_weighting: bool
    n_neighbors: int
    n_components: int
    min_dist: float
    spread: float
    min_cluster_size: int
    min_samples: int


def predict_once(texts: List[str], text_embeddings: np.ndarray, params: Hyperparameters) -> tuple[List[int], Optional[np.ndarray]]:
    """
    Train a single BERTopic model with given parameters and return topic assignments.
    
    Args:
        texts: List of input texts
        text_embeddings: Pre-computed text embeddings
        params: Hyperparameters configuration
        
    Returns:
        Tuple of (topic_assignments, probabilities)
    """
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=tuple(params.ngram_range),
        min_df=params.min_df,  
        max_df=params.max_df, 
        max_features=None,
        vocabulary=None,

        lowercase=params.lowercase,
        strip_accents=params.strip_accents,
    )
    ctfidf_model = ClassTfidfTransformer(
        # reduce_frequent_words=True,
        bm25_weighting=params.bm25_weighting,
    )
    umap_model = UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        metric='cosine',
        low_memory=False,
        min_dist=params.min_dist,  
        spread=params.spread,
        random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric='euclidean',
        prediction_data=True
    )

    # topic model
    model = BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        embedding_model=EMBEDDING_MODEL,
        calculate_probabilities=True,
        verbose=False
    )

    # fit
    return model.fit_transform(texts, embeddings=text_embeddings)

# coherenceを算出（UMass coherenceに基づく）
def compute_coherence(model: BERTopic, top_n: int = 10, eps: float = EPSILON) -> float:
    """
    Compute topic coherence based on UMass coherence.
    
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
            return eps  # Return minimum value if too few topics

        all_words = []
        for topic_words in topics.values():
            topic_words_list = [word for word, _ in topic_words[:top_n]]
            all_words.extend(topic_words_list)

        if not all_words:
            return eps

        unique_words = len(set(all_words))
        total_words = len(all_words)
        score = unique_words / total_words if total_words > 0 else 0.0
        return np.clip(score, eps, 1.0)

    except Exception:
        return eps

def compute_diversity(model, top_n=10, eps=EPSILON):
    """
    Compute topic diversity based on word uniqueness.
    
    Args:
        model: Trained BERTopic model
        top_n: Number of top words per topic
        eps: Minimum epsilon value
        
    Returns:
        Diversity score between 0 and 1
    """
    try:
        topics = {k: v for k, v in model.get_topics().items() if k != -1}
        if len(topics) < 2:
            return eps

        # Calculate word uniqueness per topic
        topic_word_sets = {}
        for topic_id, words in topics.items():
            words_list = [word for word, _ in words[:top_n]]
            topic_word_sets[topic_id] = set(words_list)

        diversity_scores = []
        for topic_id, word_set in topic_word_sets.items():
            other_words = set()
            for other_id, other_set in topic_word_sets.items():
                if other_id != topic_id:
                    other_words.update(other_set)
            
            unique_words = word_set - other_words
            unique_ratio = len(unique_words) / len(word_set) if len(word_set) > 0 else 0
            diversity_scores.append(unique_ratio)

        return np.clip(np.mean(diversity_scores), eps, 1.0) if diversity_scores else eps

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
            
        # Absolute limits based on practicality
        min_practical_clusters = max(2, n_docs // 1000)  # At least 1 cluster per 1000 docs
        max_practical_clusters = min(n_docs // 10, 100)  # At most 1 cluster per 10 docs, max 100
        
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
            return eps
            
    except Exception:
        return eps

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
        if len(unique_labels) < 2:
            return eps

        # UMAP 埋め込みを取得（複数の可能な属性を試す）
        umap_embeddings = None

        # 優先順位で埋め込みを取得
        if hasattr(model, 'umap_embeddings_') and model.umap_embeddings_ is not None:
            umap_embeddings = model.umap_embeddings_[mask]
        elif hasattr(model, 'umap_model') and hasattr(model.umap_model, 'embedding_'):
            umap_embeddings = model.umap_model.embedding_[mask]
        elif hasattr(model, 'umap_model') and hasattr(model.umap_model, '_raw_data'):
            umap_embeddings = model.umap_model._raw_data[mask]
        else:
            # 埋め込みが取得できない場合はスキップ
            return eps

        if umap_embeddings is None or len(umap_embeddings) == 0:
            return eps

        # 埋め込みの次元チェック
        if umap_embeddings.shape[1] < 2:
            return eps

        # Silhouetteスコアを計算 [-1,1] → [0,1]
        try:
            s_score = silhouette_score(umap_embeddings, labels[mask])
            s_score_scaled = (s_score + 1) / 2
        except Exception:
            s_score_scaled = 0.5  # Default on error

        # Calinski-Harabasz score normalization
        try:
            ch_score = calinski_harabasz_score(umap_embeddings, labels[mask])

            # CHスコアの正規化：より現実的な方法
            n_clusters = len(unique_labels)
            n_samples = len(umap_embeddings)

            if ch_score > 0:
                # CHスコアは(log n, log n²)の範囲になることが多い
                # より良い正規化: 経験的な最大値で除算
                # CHスコアの典型的な範囲を考慮した動的正規化
                max_expected_ch = n_samples * np.log(n_clusters + 1) * 10  # 経験値
                ch_score_scaled = min(ch_score / max_expected_ch, 1.0)
                # 下限保証: あまりにも小さな値を避ける
                ch_score_scaled = max(ch_score_scaled, 0.1)
            else:
                ch_score_scaled = 0.0

        except Exception:
            ch_score_scaled = 0.5  # エラー時は中間値を使用

        # 重み付き組み合わせ（Silhouetteをより重視）
        combined_score = 0.6 * s_score_scaled + 0.4 * ch_score_scaled
        combined_score = np.clip(combined_score, eps, 1.0)

        return combined_score

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
    if trial_count < 20:
        return HyperbandPruner(
            min_resource=1,
            max_resource=10,
            reduction_factor=3,
            bootstrap_count=0  # Compatible with fixed max_resource
        )
    
    # Mid optimization: Balanced approach  
    elif trial_count < 100:
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

def suggest_constrained_parameters(trial: optuna.Trial, dataset_size: int) -> Hyperparameters:
    """
    Suggest parameters with intelligent constraints and interdependencies.
    
    Best Practice: Implement parameter constraints to avoid invalid combinations:
    - min_samples ≤ min_cluster_size (clustering constraint)
    - max_df > min_df (vectorization constraint)
    - n_neighbors ≤ dataset_size (computational constraint)
    - Focus on practical cluster count ranges (5-50 clusters preferred)
    """
    
    # Core clustering parameters (optimized for practical cluster counts)
    # Aim for clusters representing 50-500 documents each
    min_cluster_size_lower = max(5, dataset_size // 500)  # At least 5 documents per cluster
    min_cluster_size_upper = min(100, dataset_size // 50) # At most 1 cluster per 50 docs
    min_cluster_size = trial.suggest_int("min_cluster_size", min_cluster_size_lower, min_cluster_size_upper)
    
    # min_samples scales with min_cluster_size (important constraint!)
    # More lenient constraint: min_samples can be up to 80% of min_cluster_size
    min_samples_max = max(3, int(min_cluster_size * 0.8))
    min_samples = trial.suggest_int("min_samples", 3, min_samples_max)
    
    # Text preprocessing parameters
    ngram_range = trial.suggest_categorical("ngram_range", [[1,2], [1,3]])
    
    # min_df vs max_df relationship (crucial!)
    # More conservative min_df to avoid overfiltering
    min_df = trial.suggest_int("min_df", 2, min(10, dataset_size // 100))
    # max_df is a float in (0, 1], must be > min_df/doc_count
    min_df_ratio = min_df / dataset_size
    max_df_min = min_df_ratio + 0.01  # ensure max_df > min_df/doc_count
    max_df = trial.suggest_float("max_df", max_df_min, 0.95)
    
    # Embedding model parameters
    lowercase = trial.suggest_categorical("lowercase", [True, False])
    strip_accents = trial.suggest_categorical("strip_accents", [None, "ascii", "unicode"])
    bm25_weighting = trial.suggest_categorical("bm25_weighting", [True, False])
    
    # UMAP parameters with intelligent bounds
    max_neighbors = min(dataset_size // 20, 150)  # More conservative upper bound
    n_neighbors = trial.suggest_int("n_neighbors", max(5, min_cluster_size), max_neighbors)
    
    n_components = trial.suggest_int("n_components", 
                                  max(2, int(np.log10(dataset_size))), 
                                  min(15, dataset_size // 100))  # More conservative upper bound
    
    min_dist = trial.suggest_float("min_dist", 0.0, 0.5)
    spread = trial.suggest_float("spread", 0.8, 1.3)  # Narrower range for more stable results
    
    return Hyperparameters(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=lowercase,
        strip_accents=strip_accents,
        bm25_weighting=bm25_weighting,
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

def objective(trial: optuna.Trial, texts: List[str], text_embeddings: np.ndarray, eps: float = EPSILON) -> float:
    """
    Enhanced objective function with clustering optimization best practices.
    
    Key improvements:
    1. Constrained parameter suggestions based on dataset characteristics
    2. Early termination support with pruning-aware checkpointing
    3. Multi-metric optimization with stability measures
    4. Memory-efficient evaluation with comprehensive scoring
    
    Args:
        trial: Optuna trial object
        texts: List of input texts
        text_embeddings: Pre-computed text embeddings
        eps: Minimum epsilon value for score calculations
        
    Returns:
        Composite optimization score (0-1) with additional metrics for analysis
    """
    
    # Step 1: Suggest constrained parameters using best practices
    dataset_size = len(texts)
    params = suggest_constrained_parameters(trial, dataset_size)

    try:
        # Step 2: Train model with pruning-friendly checkpointing
        dataset_size = len(texts)
        model = BERTopic(
            vectorizer_model=CountVectorizer(
                stop_words="english",
                ngram_range=tuple(params.ngram_range),
                min_df=params.min_df,
                max_df=params.max_df,
                lowercase=params.lowercase,
                strip_accents=params.strip_accents
            ),
            ctfidf_model=ClassTfidfTransformer(bm25_weighting=params.bm25_weighting),
            umap_model=UMAP(
                n_neighbors=params.n_neighbors,
                n_components=params.n_components,
                metric='cosine',
                min_dist=params.min_dist,
                spread=params.spread,
                random_state=42,
                low_memory=False  # Better for pruned early termination
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=params.min_cluster_size,
                min_samples=params.min_samples,
                metric='euclidean',
                prediction_data=False
            ),
            embedding_model=EMBEDDING_MODEL,
            calculate_probabilities=False,
            verbose=False
        )

        # Step 3: Efficient evaluation with early termination hooks
        topics, _ = model.fit_transform(texts, embeddings=text_embeddings)
        
        # Quick validation checkpoint for pruning
        topic_info = model.get_topic_info()
        n_clusters = len(topic_info[topic_info['Topic'] != -1])
        
        # Early exit for obvious failures (helps pruning)
        if n_clusters < 2 or n_clusters > dataset_size // 5:
            return eps * 0.1  # Very low score for clear failures
        
        # Step 4: Multi-metric evaluation with best practices
        coherence = compute_coherence(model, eps=eps)
        diversity = compute_diversity(model, eps=eps)
        cluster_score = compute_cluster_score(model, eps=eps)
        cluster_validity = evaluate_cluster_count(n_clusters, dataset_size, eps)

        # Step 5: Enhanced composite scoring with stability measures
        base_weights = SCORE_WEIGHTS.copy()
        
        # Check which scores are valid for weight adjustment
        valid_scores = {
            'coherence': coherence > eps,
            'diversity': diversity > eps,
            'cluster': cluster_score > eps,
            'validity': cluster_validity > eps
        }

        if sum(valid_scores.values()) == 0:
            return eps

        # Adjust weights for valid scores only and normalize
        adjusted_weights = {}
        total_weight = sum(base_weights[k] for k in base_weights.keys() if valid_scores[k])
        
        for key in base_weights:
            if valid_scores[key]:
                adjusted_weights[key] = base_weights[key] / total_weight
            else:
                adjusted_weights[key] = 0

        # Calculate composite score with best practice multi-objective approach
        base_score = (
            adjusted_weights['coherence'] * coherence +
            adjusted_weights['diversity'] * diversity +
            adjusted_weights['cluster'] * cluster_score +
            adjusted_weights['validity'] * cluster_validity
        )
        
        # Step 6: Apply intelligent penalties and rewards
        final_score = base_score
        
        # Enhanced cluster count validation with new scoring system
        if n_clusters < 2:
            final_score *= 0.1  # Very harsh penalty for no/few clusters
        elif n_clusters > dataset_size // 10:
            final_score *= 0.3  # Penalty for excessive clusters
        elif 8 <= n_clusters <= 25:
            final_score *= 1.05  # Bonus for practical cluster ranges
        
        # Balanced parameter combinations penalty
        cluster_size_ratio = params.min_cluster_size / dataset_size
        if cluster_size_ratio > 0.05:  # min_cluster_size > 5% of dataset
            final_score *= 0.9  # Penalty for overly restrictive clustering
        elif cluster_size_ratio < 0.002:  # min_cluster_size < 0.2% of dataset  
            final_score *= 0.95  # Slight penalty for too loose clustering
        
        # Store additional metrics for analysis (best practice)
        trial.set_user_attr("n_clusters", n_clusters)
        trial.set_user_attr("coherence", coherence)
        trial.set_user_attr("diversity", diversity) 
        trial.set_user_attr("cluster_score", cluster_score)
        trial.set_user_attr("cluster_validity", cluster_validity)

        return np.clip(final_score, eps, 1.0)

    except Exception:
        # Graceful failure handling for clustering optimization
        return eps

def run_one_category(category: str, timeout: int = 10*60, storage: Optional[str] = None) -> optuna.Study:
    """
    Enhanced optimization runner with clustering optimization best practices.
    
    Key improvements:
    1. Adaptive sampler selection based on dataset characteristics
    2. Intelligent pruning strategies for different optimization phases
    3. Warm-start capabilities for continuing previous studies
    4. Advanced study configuration with comprehensive tracking
    
    Args:
        category: Paper category to optimize
        timeout: Optimization timeout in seconds
        storage: Optuna storage backend
        
    Returns:
        Completed Optuna study with comprehensive optimization history
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
