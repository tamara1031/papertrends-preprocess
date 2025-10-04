from typing import List, Tuple, Any
from dataclasses import dataclass

import gc
import pickle

import torch
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

EMBEDDING_MODEL = get_custom_embedding_model()

def get_papers(category: str) -> List[Paper]:
    with open(f"./preprocessed/{category}/papers.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def get_text_embeddings(category: str) -> np.ndarray:
    with open(f"./preprocessed/{category}/text_embeddings.npy", "rb") as f:
        embeddings = np.load(f)
    return embeddings

@dataclass
class Params:
    ngram_range: Tuple[int, int]
    min_df: float | int
    max_df: float | int
    lowercase: bool
    strip_accents: Any | None
    bm25_weighting: bool
    n_neighbors: int
    n_components: int
    min_dist: float
    spread: float
    min_cluster_size: int
    min_samples: int


def predict_once(texts, text_embeddings, params: Params):
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=params.ngram_range,
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
def compute_coherence(model, top_n=10, eps=1e-6):
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

def compute_diversity(model, top_n=10, eps=1e-6):
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

def compute_cluster_score(model, eps=1e-6):
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

def objective(trial: optuna.Trial, texts, text_embeddings, eps=1e-6):

    params = Params(
        ngram_range=trial.suggest_categorical("ngram_range", [(1,1), (1,2), (1,3)]),
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
                ngram_range=params.ngram_range,
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
                prediction_data=True
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

        # 詳細なログ出力（デバッグ用）
        print(f"  Coherence: {coherence:.4f}")
        print(f"  Diversity: {diversity:.4f}")
        print(f"  Cluster Score: {cluster_score:.4f}")

        # 適応的な重み付け：各スコアの信頼性に基づいて重みを調整
        base_weights = {
            'coherence': 0.5,    # 最も重要な指標
            'diversity': 0.3,    # 重要だがcoherenceほどではない
            'cluster': 0.2       # 補助的な指標
        }

        # 各スコアが有効かチェックして重みを調整
        valid_scores = {
            'coherence': coherence > eps,
            'diversity': diversity > eps,
            'cluster': cluster_score > eps
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
            adjusted_weights['cluster'] * cluster_score
        )

        print(f"  Final combined score: {score:.4f}")
        return score

    except Exception as e:
        print(f"Trial failed: {e}")
        import traceback
        print(f"Trial traceback: {traceback.format_exc()}")
        return eps

def run_for_category(category: str, n_trials: int = 100):

    papers = get_papers(category)
    text_embeddings = get_text_embeddings(category)
    texts = [EMBEDDING_MODEL.get_input_text(paper) for paper in papers]
    del papers
    gc.collect()

    study = optuna.create_study(
        direction="maximize",
        study_name="search_params",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, texts, text_embeddings),
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    return study

if __name__ == "__main__":
    study = run_for_category("physics.geo-ph", n_trials=100)
    print("Study completed!")
    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")
    print(f"Number of finished trials: {len(study.trials)}")