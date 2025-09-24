from typing import List

from datetime import datetime

from tqdm import tqdm

import os
import gc
import pickle
import numpy as np
import json
import pandas as pd
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

from common.utils import get_custom_embedding_model, get_category_codes

VERSION = "0.0.2"

def already_processed(category: str):
    return os.path.exists(f"./visualizations/{category}.json")

def load_model(category: str):
    embedding_model = get_custom_embedding_model()
    model = BERTopic.load(f"./models/{category}", embedding_model=embedding_model)
    return model

def load_papers(category: str):
    with open(f"./preprocessed/{category}/papers.pkl", "rb") as f:
        papers = pickle.load(f)
    return papers

def load_representative_docs(category: str):
    with open(f"./models/{category}/representative_docs.pkl", "rb") as f:
        representative_docs = pickle.load(f)
    return representative_docs

def convert_representative_docs_to_dataframe(representative_docs):
    # 各トピックの代表文書（文字列配列）を展開してDataFrameに変換
    rows = []
    for topic_id, docs in representative_docs.items():
        # docsは文字列の配列

        reps = []
        for doc_text in docs:
            # 各文字列は "title[SEP]abstract" の形式
            reps.append(doc_text)


        rows.append({
            'Topic': int(topic_id),
            'Representative_Docs': reps,
        })
    
    return pd.DataFrame(rows)

def get_topic_distr(category: str, model: BERTopic, texts: List[str]):
    """データを読み込み、トピック分布を計算する"""
    
    # トピック分布を計算
    # tmp_path = f"./visualizations/{category}/topic_distr_tmp.npy"
    # os.makedirs(f"./visualizations/{category}", exist_ok=True)
    # if os.path.exists(tmp_path):
    #     topic_distr = np.load(tmp_path)
    # else:
    topic_distr, _ = model.approximate_distribution(
        texts,
        calculate_tokens=True,
        # use_embedding_model=True,
        # separator=model.embedding_model.tokenizer.sep_token
    )
    #     np.save(tmp_path, topic_distr)

    return topic_distr

def calculate_topic_correlations(document_info: pd.DataFrame):
    """
    トピック間の相関行列を計算
    Args:
        document_info: 各文書の情報を含むDataFrame（"Topic_Distribution"カラムに分布が格納されていること）
    Returns:
        correlation_matrix: トピック間の相関行列 (n_topics, n_topics)
    """
    # "Topic_Distribution"カラムから分布行列を作成
    # 各要素はnp.ndarrayまたはlistなので、2次元配列に変換
    topic_distr = np.vstack(document_info["Topic_Distribution"].values)
    n_topics = topic_distr.shape[1]
    correlation_matrix = np.zeros((n_topics, n_topics))

    # コサイン類似度
    cosine_sim = cosine_similarity(topic_distr.T)

    # Jensen-Shannon divergence（距離を相関に変換）
    js_corr = np.zeros((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(n_topics):
            if i == j:
                js_corr[i, j] = 1.0
            else:
                js_dist = jensenshannon(topic_distr[:, i], topic_distr[:, j])
                js_corr[i, j] = 1.0 - js_dist

    # コサイン類似度を[-1, 1]に変換: 2 * cosine - 1
    cosine_normalized = 2 * cosine_sim - 1

    # JS divergenceを[-1, 1]に変換: 2 * js_corr - 1
    js_normalized = 2 * js_corr - 1

    # 各手法の重み付き平均
    weights = {
        'cosine': 0.3,
        'js_divergence': 0.7
    }

    correlation_matrix = (
        weights['cosine'] * cosine_normalized +
        weights['js_divergence'] * js_normalized
    )

    # 対角成分を1.0に設定（自己相関）
    np.fill_diagonal(correlation_matrix, 1.0)

    # 相関値を[-1, 1]の範囲にクリップ
    correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)

    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Correlation range: [{np.min(correlation_matrix):.3f}, {np.max(correlation_matrix):.3f}]")

    return correlation_matrix

def generate_topics_data(topic_info: pd.DataFrame):
    """トピックデータを生成（新しいJSON構造に対応）"""
    print("Generating topics data...")
    
    # 各トピックのデータを生成
    topics_data = {}
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]

        name = row["Name"]
        # "Representation"カラムが存在する場合はそれを使う
        words = row["Keyword"] if "Keyword" in row else []
        # "Count"カラムが存在する場合はそれを使う
        paper_count = int(row["Count"]) if "Count" in row else 0

        topics_data[str(topic_id)] = {
            "name": name, 
            "keywords": words,
            "count": paper_count
        }
    
    return topics_data


def generate_series_data(document_info: pd.DataFrame):
    """時系列データを生成（月ごとに集計, keyは%Y-%m）"""
    print("Generating series data...")

    # 論文リストを取得
    papers = document_info["Paper"].tolist()
    # 各論文の主トピック（argmax）を計算
    topic_distr = np.vstack(document_info["Topic_Distribution"].values)
    main_topics = np.argmax(topic_distr, axis=1)
    n_topics = topic_distr.shape[1]

    # 月ごと・トピックごとにカウント
    series_data = {}
    for idx, paper in enumerate(papers):
        month_key = paper.published.strftime("%Y-%m")
        topic_id = main_topics[idx]
        if month_key not in series_data:
            series_data[month_key] = [0] * n_topics
        series_data[month_key][topic_id] += 1

    # 月順にソートして返す
    sorted_series_data = {month: series_data[month] for month in sorted(series_data.keys())}
    return sorted_series_data


def get_representative_papers_by_topic(topic_info: pd.DataFrame, document_info: pd.DataFrame, n_papers_per_topic: int = 5):
    """各トピックの代表論文を取得（topic_infoのRepresentative_Docsからdocument_info内で検索）"""
    print("Getting representative papers by topic...")

    # document_infoのPaper列から論文情報を取得
    papers = document_info["Paper"].tolist()
    # 論文テキスト（title[SEP]abstract）を作成してインデックス化
    paper_text_to_index = {}
    for idx, paper in enumerate(papers):
        # "title[SEP]abstract" 形式
        key = f"{paper.title}[SEP]{paper.abstract}"
        paper_text_to_index[key] = idx

    papers_by_topic = {}

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        rep_docs = row.get("Representative_Docs", [])
        topic_papers = []
        count = 0
        for doc_text in rep_docs:
            idx = paper_text_to_index.get(doc_text)
            if idx is not None and idx < len(papers):
                paper = papers[idx]
                paper_info = {
                    "title": paper.title,
                    "year": paper.published.strftime("%Y-%m"),
                    "abstract": paper.abstract,
                    "arxiv_id": paper.arxiv_id
                }
                topic_papers.append(paper_info)
                count += 1
                if count >= n_papers_per_topic:
                    break
        papers_by_topic[str(topic_id)] = topic_papers

    return papers_by_topic

def generate_visualization_data(topic_info: pd.DataFrame, document_info: pd.DataFrame):

    # 各トピックのデータを生成
    topics_data = generate_topics_data(topic_info)
    
    # 相関行列を計算
    correlation_matrix = calculate_topic_correlations(document_info)
    
    # 時系列データを生成
    series_data = generate_series_data(document_info)

    # 代表論文を取得
    papers_data = get_representative_papers_by_topic(topic_info, document_info)

    # メタデータを生成
    metadata = generate_metadata(document_info)
    
    return {
        "topics": {
            "data": topics_data,
            "correlations": correlation_matrix.tolist(),
            "series": series_data,
            "papers": papers_data
        },
        "metadata": metadata
    }


def generate_metadata(document_info: pd.DataFrame = None):
    """メタデータを生成（集計期間を含む）"""
    metadata = {
        "lastUpdated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataVersion": VERSION
    }
    # 集計期間を追加
    papers = document_info["Paper"].tolist()
    published_dates = [paper.published for paper in papers]

    start_date = min(published_dates).strftime("%Y-%m")
    end_date = max(published_dates).strftime("%Y-%m")
    metadata["period"] = {
        "start": start_date,
        "end": end_date
    }
    return metadata


def save_visualization_data(category, visualization_data):
    """可視化データをJSONファイルとして保存"""
    
    # 出力ディレクトリを作成
    output_dir = f"./visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # JSONファイルとして保存
    output_path = f"{output_dir}/{category}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(visualization_data
    , f, ensure_ascii=False, indent=2)
    
    print(f"Visualization data saved to: {output_path}")
    return output_path

def process_category(category):

    gc.collect()

    if already_processed(category):
        return

    # データを読み込み
    model = load_model(category)
    papers = load_papers(category)
    representative_docs = load_representative_docs(category)

    # create topic_info
    topic_info = model.get_topic_info()
    representative_docs_df = convert_representative_docs_to_dataframe(representative_docs)

    topic_info = topic_info.drop(columns=["Representative_Docs"], errors="ignore").merge(
        representative_docs_df, how="left", left_on="Topic", right_on="Topic"
    )

    # get_topics()の出力から各トピックごとに寄与度の高い単語10個を抽出し、topic_infoに"Keywords"列として追加
    topics_dict = model.get_topics()
    # topics_dict: {topic_id: [(word, score), ...], ...}
    top10_words_map = {}
    for topic_id, words_scores in topics_dict.items():
        # 上位10単語を抽出
        top_words = [(word, score) for word, score in words_scores[:10]]
        # topic_idを整数に変換してキーとして使用
        top10_words_map[int(topic_id)] = top_words

    # topic_infoの"Topic"列に対応する"Keywords"列を追加
    topic_info["Keyword"] = topic_info["Topic"].map(top10_words_map)
    
    # マッチしないトピック（None値）を空のリストで埋める
    topic_info["Keyword"] = topic_info["Keyword"].fillna("").apply(lambda x: [] if x == "" else x)

    topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True) # Topicが-1の場合を除外

    # create document_info
    texts = [model.embedding_model.get_input_text(paper) for paper in papers]
    document_info = model.get_document_info(texts)
    topic_distr = get_topic_distr(category, model, texts)

    document_info["Topic_Distribution"] = [dist for dist in topic_distr]
    # document_infoのindexとpapersのindexを対応させて、paper情報をdocument_infoに追加
    document_info["Paper"] = [papers[idx] for idx in document_info.index]

    # トピックデータを生成
    visualization_data = generate_visualization_data(topic_info, document_info)
    
    
    # 可視化データをJSONファイルとして保存
    output_path = save_visualization_data(category, visualization_data)
    
    # 結果の概要を表示
    print(f"\n=== Visualization Data Generated ===")
    print(f"Category: {category}")
    print(f"Output file: {output_path}")
    
    return output_path

def main():
    """メイン処理: データ取得 → 可視化"""
    categories = get_category_codes()

    for category in tqdm(categories, desc="Processing categories"):
        process_category(category)


if __name__ == "__main__":
    main()