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
            return eps  # トピックが少なくても最小値を返す

        # 文書-単語行列を取得（モデルから抽出）
        vectorizer = model.vectorizer_model
        if not hasattr(model, 'fit_transform') or not hasattr(model, 'vectorizer_model'):
            return eps

        # 訓練データから文書-単語行列を再構築
        texts = model.get_document_info(model.get_document_ids()) if hasattr(model, 'get_document_ids') else []
        if len(texts) == 0:
            # 代替としてc-TF-IDFの情報を利用
            ctfidf_matrix = model.c_tf_idf_
            if ctfidf_matrix is None or ctfidf_matrix.shape[0] == 0:
                return eps

            # 簡易的なcoherence計算（単語重複を避ける）
            all_words = []
            for topic_id, words in topics.items():
                topic_words = [word for word, _ in words[:top_n]]
                all_words.extend(topic_words)

            if len(all_words) == 0:
                return eps

            # 単語の多様性に基づく簡易coherence
            unique_words = len(set(all_words))
            total_words = len(all_words)
            coherence_score = unique_words / total_words if total_words > 0 else 0.0
            return np.clip(coherence_score, eps, 1.0)

        try:
            # テキストから文書-単語行列を作成
            doc_term_matrix = vectorizer.transform(texts)
            word_probabilities = doc_term_matrix.sum(axis=0).A1 / doc_term_matrix.sum()

            # 各トピックについてcoherenceを計算
            coherence_scores = []
            for topic_id, words in topics.items():
                topic_words = [word for word, _ in words[:top_n]]
                if len(topic_words) < 2:
                    continue

                # 各単語ペアについてcoherenceを計算（UMass方式）
                topic_coherence = 0.0
                pair_count = 0

                for i, word_i in enumerate(topic_words):
                    for j in range(i + 1, min(i + 6, len(topic_words))):  # 上位5ペアのみ
                        word_j = topic_words[j]

                        # 単語のインデックスを取得
                        try:
                            idx_i = vectorizer.vocabulary_.get(word_i)
                            idx_j = vectorizer.vocabulary_.get(word_j)

                            if idx_i is None or idx_j is None:
                                continue

                            # P(w_j)の計算
                            p_wj = word_probabilities[idx_j]

                            # P(w_i, w_j)の計算（共起確率）
                            # 簡易版：両方の単語を含む文書の割合
                            docs_with_i = doc_term_matrix[:, idx_i].toarray().flatten()
                            docs_with_j = doc_term_matrix[:, idx_j].toarray().flatten()

                            co_occur = np.sum(docs_with_i & docs_with_j)
                            total_docs = doc_term_matrix.shape[0]

                            p_wi_wj = co_occur / total_docs if total_docs > 0 else eps

                            # UMass coherence計算
                            if p_wj > 0:
                                pair_coherence = np.log((p_wi_wj + eps) / p_wj)
                                topic_coherence += pair_coherence
                                pair_count += 1

                        except (KeyError, IndexError):
                            continue

                if pair_count > 0:
                    coherence_scores.append(topic_coherence / pair_count)

            # 全トピックの平均coherenceを返す
            if len(coherence_scores) == 0:
                return eps

            avg_coherence = np.mean(coherence_scores)
            # 負の値をクリップして0-1の範囲に正規化
            coherence_score = np.clip((avg_coherence + 2) / 4, eps, 1.0)  # -2〜2を0〜1にマッピング
            return coherence_score

        except Exception as inner_e:
            print(f"Coherence calculation inner failed: {inner_e}")
            # フォールバック：単純な多様性ベースの計算
            all_words = []
            for topic_id, words in topics.items():
                topic_words = [word for word, _ in words[:top_n]]
                all_words.extend(topic_words)

            if len(all_words) == 0:
                return eps

            unique_words = len(set(all_words))
            total_words = len(all_words)
            coherence_score = unique_words / total_words if total_words > 0 else 0.0
            return np.clip(coherence_score, eps, 1.0)

    except Exception as e:
        print(f"Coherence calculation failed: {e}")
        import traceback
        print(f"Coherence traceback: {traceback.format_exc()}")
        return eps

def compute_diversity(model, top_n=10, eps=EPSILON):
    try:
        topics = {k: v for k, v in model.get_topics().items() if k != -1}
        if len(topics) < 2:
            return eps

        # 各トピックの単語セットを作成
        topic_word_sets = {}
        for topic_id, words in topics.items():
            topic_words = [word for word, _ in words[:top_n]]
            topic_word_sets[topic_id] = set(topic_words)

        if len(topic_word_sets) < 2:
            return eps

        # 単語の独自性に基づく多様性計算
        all_unique_words = set()
        topic_unique_words = {}

        for topic_id, word_set in topic_word_sets.items():
            all_unique_words.update(word_set)
            # このトピック独自の単語をカウント
            other_words = set()
            for other_id, other_set in topic_word_sets.items():
                if other_id != topic_id:
                    other_words.update(other_set)
            unique_words = word_set - other_words
            topic_unique_words[topic_id] = unique_words

        # 各トピックが独自に持つ単語の割合を計算
        diversity_scores = []
        for topic_id in topic_unique_words:
            unique_count = len(topic_unique_words[topic_id])
            total_count = len(topic_word_sets[topic_id])
            if total_count > 0:
                diversity_scores.append(unique_count / total_count)

        if len(diversity_scores) == 0:
            return eps

        avg_diversity = np.mean(diversity_scores)

        # より洗練された多様性評価：トピックベクトル間の距離も考慮
        try:
            # c-TF-IDFベクトルを取得してトピック間の距離を計算
            if hasattr(model, 'c_tf_idf_') and model.c_tf_idf_ is not None:
                ctfidf_matrix = model.c_tf_idf_
                if ctfidf_matrix.shape[0] > 1:
                    # コサイン距離を計算して多様性を評価
                    from sklearn.metrics.pairwise import cosine_distances

                    # 有効なトピックのみを選択
                    valid_topic_ids = list(topics.keys())
                    if len(valid_topic_ids) > 1:
                        topic_indices = [i for i, tid in enumerate(valid_topic_ids) if tid in topics]
                        if len(topic_indices) > 1:
                            topic_vectors = ctfidf_matrix[topic_indices]

                            # コサイン距離を計算
                            cosine_dist = cosine_distances(topic_vectors)

                            # 多様性は距離の平均値として計算（高いほど良い）
                            distance_score = np.mean(cosine_dist[np.triu_indices_from(cosine_dist, k=1)])

                            # 独自性スコアと距離スコアを組み合わせ
                            combined_diversity = 0.6 * avg_diversity + 0.4 * distance_score
                            combined_diversity = np.clip(combined_diversity, eps, 1.0)
                            return combined_diversity
        except Exception:
            # c-TF-IDFが利用できない場合は独自性スコアのみを使用
            pass

        diversity_score = np.clip(avg_diversity, eps, 1.0)
        return diversity_score

    except Exception as e:
        print(f"Diversity calculation failed: {e}")
        import traceback
        print(f"Diversity traceback: {traceback.format_exc()}")
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
            
    except Exception as e:
        print(f"Cluster count evaluation failed: {e}")
        import traceback
        print(f"Cluster count traceback: {traceback.format_exc()}")
        return eps

def compute_cluster_score(model, eps=EPSILON):
    """
    BERTopic の UMAP 埋め込みを使ったクラスタリング評価
    Silhouette + Calinski-Harabasz を組み合わせ、適正に正規化
    """
    try:
        # HDBSCAN ラベルを取得
        if not hasattr(model, 'hdbscan_model') or model.hdbscan_model is None:
            return eps

        labels = model.hdbscan_model.labels_
        if labels is None or len(labels) == 0:
            return eps

        mask = labels != -1  # ノイズを除外

        # 有効クラスタが2つ以上ない場合は最小値返却
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
            s_score_scaled = 0.5  # エラー時は中間値を使用

        # Calinski-Harabaszスコアをより適切に正規化
        try:
            ch_score = calinski_harabasz_score(umap_embeddings, labels[mask])

            # CHスコアの正規化：クラスタ数に応じて調整
            n_clusters = len(unique_labels)
            n_samples = len(umap_embeddings)

            # CHスコアは一般的に大きな値ほど良いが、クラスタ数に依存する
            # ここでは単純に0-1にクリップ
            if ch_score > 0:
                # 一般的なCHスコアの範囲に基づいて正規化（経験値に基づく）
                ch_score_scaled = min(ch_score / (n_samples * n_clusters), 1.0)
            else:
                ch_score_scaled = 0.0

        except Exception:
            ch_score_scaled = 0.5  # エラー時は中間値を使用

        # 重み付き組み合わせ（Silhouetteをより重視）
        combined_score = 0.6 * s_score_scaled + 0.4 * ch_score_scaled
        combined_score = np.clip(combined_score, eps, 1.0)

        return combined_score

    except Exception as e:
        print(f"Cluster score calculation failed: {e}")
        import traceback
        print(f"Cluster score traceback: {traceback.format_exc()}")
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

        # Detailed logging output for debugging
        print(f"  Coherence: {coherence:.4f}")
        print(f"  Diversity: {diversity:.4f}")
        print(f"  Cluster Score: {cluster_score:.4f}")
        print(f"  N Clusters: {n_clusters}")
        print(f"  Cluster Validity: {cluster_validity:.4f}")

        # Use predefined score weights for consistency
        base_weights = SCORE_WEIGHTS.copy()

        # Check which scores are valid for weight adjustment
        valid_scores = {
            'coherence': coherence > eps,
            'diversity': diversity > eps,
            'cluster': cluster_score > eps,
            'validity': cluster_validity > eps
        }

        valid_count = sum(valid_scores.values())
        print(f"  Valid scores count: {valid_count}")

        if valid_count == 0:
            print("  All scores invalid, returning eps")
            return eps

        # 有効なスコアのみで重みを再計算
        adjusted_weights = {}
        total_weight = 0

        for key in base_weights:
            if valid_scores[key]:
                adjusted_weights[key] = base_weights[key]
                total_weight += base_weights[key]
            else:
                adjusted_weights[key] = 0
                print(f"  {key} score invalid, weight set to 0")

        # 重みの正規化
        if total_weight > 0:
            for key in adjusted_weights:
                adjusted_weights[key] /= total_weight
                print(f"  Adjusted weight for {key}: {adjusted_weights[key]:.4f}")

        # 最終スコアの計算
        score = (
            adjusted_weights['coherence'] * coherence +
            adjusted_weights['diversity'] * diversity +
            adjusted_weights['cluster'] * cluster_score +
            adjusted_weights['validity'] * cluster_validity
        )

        print(f"  Final combined score: {score:.4f}")
        return score

    except Exception as e:
        print(f"Trial failed: {e}")
        import traceback
        print(f"Trial traceback: {traceback.format_exc()}")
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

