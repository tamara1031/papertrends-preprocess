from typing import List, Tuple, Any, Optional, Dict, Union
from dataclasses import dataclass

import gc, os
import pickle
import json

import numpy as np

import optuna
from bertopic import BERTopic

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model, get_category_codes
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# Constants
EPSILON = 1e-6
EMBEDDING_MODEL = get_custom_embedding_model()

# Dataset size thresholds for adaptive parameter ranges
SMALL_DATASET_THRESHOLD = 5000
MEDIUM_DATASET_THRESHOLD = 50000

# Score weights for composite evaluation
SCORE_WEIGHTS = {
    'coherence': 0.35,
    'diversity': 0.25,
    'cluster': 0.30,
    'validity': 0.10
}

# Cluster count validation thresholds
MIN_VALID_CLUSTERS = 3

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
        min_df=params.min_df,  # 0.0001%以上に出現（最低2件, 最高30件）
        max_df=params.max_df, # modelsなどを弾きたい
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
    Evaluate the appropriateness of cluster count based on dataset size and characteristics.
    
    This function applies multiple validation criteria to assess if the number of
    clusters is appropriate for the given dataset size. It considers:
    - Square root rule (√n rule) for small datasets
    - Elbow method approximation for medium datasets  
    - Log-based scaling for large datasets
    - Domain-specific constraints (minimum viable clusters, maximum interpretability)
    
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
            return eps * 0.5
            
        # Too many clusters relative to dataset size
        max_reasonable_clusters = min(n_docs // 5, n_docs * 0.3)
        if n_clusters > max_reasonable_clusters:
            return eps
            
        # Calculate expected optimal cluster ranges based on dataset size
        if n_docs <= SMALL_DATASET_THRESHOLD:
            # Small datasets: Use square root rule with adaptations
            expected_clusters_sqrt = int(np.sqrt(n_docs))
            min_reasonable = max(2, int(expected_clusters_sqrt * 0.5))
            max_reasonable = int(expected_clusters_sqrt * 2.0)
            
        elif n_docs <= MEDIUM_DATASET_THRESHOLD:
            # Medium datasets: Use modified Gap statistic approximation
            expected_clusters_elbow = int(np.sqrt(n_docs / 2))
            min_reasonable = max(3, int(expected_clusters_elbow * 0.4))
            max_reasonable = int(expected_clusters_elbow * 1.8)
            
        else:
            # Large datasets: Use log-based scaling
            expected_clusters_log = int(np.log2(n_docs) * 4)
            min_reasonable = max(10, int(expected_clusters_log * 0.3))
            max_reasonable = int(expected_clusters_log * 1.5)
        
        # Check if within reasonable range
        if n_clusters < min_reasonable or n_clusters > max_reasonable:
            # Calculate penalty based on distance from optimal range  
            optimal_center = (min_reasonable + max_reasonable) / 2
            distance_from_optimal = abs(n_clusters - optimal_center) / max(optimal_center, 1)
            # Use gentler exponential penalty
            return max(eps, np.exp(-distance_from_optimal * 1.5))
            
        # Calculate score based on optimal range proximity
        if n_docs <= SMALL_DATASET_THRESHOLD:
            # Fine-grained scoring for small datasets
            expected_clusters_sqrt = int(np.sqrt(n_docs))
            distance = abs(n_clusters - expected_clusters_sqrt) / max(expected_clusters_sqrt, 1)
            
            # Normalize distance and apply gentler penalty
            normalized_distance = min(distance, 1.0)
            return max(eps, 1.0 - normalized_distance * 0.3)
            
        elif n_docs <= MEDIUM_DATASET_THRESHOLD:
            # Moderate tolerance for medium datasets
            expected_clusters_elbow = int(np.sqrt(n_docs / 2))
            distance = abs(n_clusters - expected_clusters_elbow) / max(expected_clusters_elbow, 1)
            
            # Apply moderate penalty with cap
            normalized_distance = min(distance, 1.0)
            return max(eps, 1.0 - normalized_distance * 0.2)
            
        else:
            # Flexible scoring for large datasets (interpretability matters)
            expected_clusters_log = int(np.log2(n_docs) * 4)
            distance = abs(n_clusters - expected_clusters_log) / max(expected_clusters_log, 1)
            
            # Large datasets can have more clusters naturally
            normalized_distance = min(distance, 1.0)
            return max(eps, 1.0 - normalized_distance * 0.25)
            
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

def objective(trial: optuna.Trial, texts: List[str], text_embeddings: np.ndarray, eps: float = EPSILON) -> float:
    """
    Optuna objective function for BERTopic hyperparameter optimization.
    
    This function trains a BERTopic model with suggested hyperparameters and
    returns an evaluation score based on coherence, diversity, clustering quality,
    and cluster count validity.
    
    Args:
        trial: Optuna trial object
        texts: List of input texts
        text_embeddings: Pre-computed text embeddings
        eps: Minimum epsilon value for score calculations
        
    Returns:
        Composite optimization score (0-1)
    """

    params = Hyperparameters(
        ngram_range=trial.suggest_categorical("ngram_range", [[1,1], [1,2], [1,3]]),
        min_df=trial.suggest_int("min_df", 2, 20),
        max_df=trial.suggest_float("max_df", 0.2, 0.95),  # 0.3-0.80の範囲に修正
        lowercase=trial.suggest_categorical("lowercase", [True, False]),
        strip_accents=trial.suggest_categorical("strip_accents", [None, "ascii", "unicode"]),
        bm25_weighting=trial.suggest_categorical("bm25_weighting", [True, False]),
        n_neighbors=trial.suggest_int("n_neighbors", 5, 50),
        n_components=trial.suggest_int("n_components", 2, 20),
        min_dist=trial.suggest_float("min_dist", 0.0, 0.8),
        spread=trial.suggest_float("spread", 0.8, 2.0),
        min_cluster_size=trial.suggest_int("min_cluster_size", 10, 100),  # 範囲を絞る
        min_samples=trial.suggest_int("min_samples", 5, 50),  # 範囲を絞る
    )

    try:
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
                random_state=42
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

        # infer
        topics, probs = model.fit_transform(texts, embeddings=text_embeddings)

        # evaluate
        coherence = compute_coherence(model, eps=eps)
        diversity = compute_diversity(model, eps=eps)
        cluster_score = compute_cluster_score(model, eps=eps)
        
        # Calculate cluster count validity
        topic_info = model.get_topic_info()
        n_clusters = len(topic_info[topic_info['Topic'] != -1])
        cluster_validity = evaluate_cluster_count(n_clusters, len(texts), eps)

        # Calculate composite score using all valid metrics
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

        # Calculate composite score
        score = (
            adjusted_weights['coherence'] * coherence +
            adjusted_weights['diversity'] * diversity +
            adjusted_weights['cluster'] * cluster_score +
            adjusted_weights['validity'] * cluster_validity
        )
        
        # Apply penalty for extreme cluster counts
        if n_clusters < MIN_VALID_CLUSTERS or n_clusters > int(len(texts) * 0.3):
            score *= 0.5

        return score

    except Exception:
        return eps

def run_one_category(category: str, timeout: int = 10*60, storage: Optional[str] = None) -> optuna.Study:
    """
    Run hyperparameter optimization for a single category.
    
    Args:
        category: Paper category to optimize
        timeout: Optimization timeout in seconds
        storage: Optuna storage backend
        
    Returns:
        Completed Optuna study
    """

    papers = get_papers(category)
    text_embeddings = get_text_embeddings(category)
    texts = [EMBEDDING_MODEL.get_input_text(paper) for paper in papers]
    del papers
    gc.collect()

    study = optuna.create_study(
        storage=storage,
        direction="maximize",
        study_name="search_params",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, texts, text_embeddings),
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    return study


if __name__ == "__main__":
    category = "physics.geo-ph"

    model_path = f"./models/{category}"
    os.makedirs(model_path, exist_ok=True)

    study_storage_path = f"sqlite:///{model_path}/search_params.db"
    study = run_one_category("physics.geo-ph", timeout=5*60, storage=study_storage_path)

    params_storage_path = f"{model_path}/best_params.json"
    with open(params_storage_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

