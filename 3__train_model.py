from typing import List

import os

import json
from tqdm import tqdm
import gc
import pickle

import torch
import numpy as np

from bertopic import BERTopic

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance

from common.domain.dto import Paper
from common.utils import get_custom_embedding_model, get_category_codes

def get_papers(category: str) -> List[Paper]:
    with open(f"./preprocessed/{category}/papers.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def get_text_embeddings(category: str) -> np.ndarray:
    with open(f"./preprocessed/{category}/text_embeddings.npy", "rb") as f:
        embeddings = np.load(f)
    return embeddings

# 一旦固定でAIカテゴリを処理
categories = get_category_codes()
# categories = ["cs.IR"]

for category in tqdm(categories, desc="Processing categories"):

    model_dir = f"./models/{category}"
    if os.path.exists(model_dir):
        continue
    os.makedirs(model_dir, exist_ok=True)

    # 前処理済データを取得
    papers = get_papers(category)
    text_embeddings = get_text_embeddings(category)
    embedding_model = get_custom_embedding_model()
    texts = [embedding_model.get_input_text(paper) for paper in papers]

    # KeyBERTでキーワードを抽出
    del papers
    gc.collect()

    # Clustering using vocab
    top_n_words = 15

    num_texts = len(texts)
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=min(30, max(2, int(num_texts * 0.0001))),  # 0.0001%以上に出現（最低2件, 最高30件）
        max_df=int(num_texts * 0.65), # modelsなどを弾きたい
        max_features=None,
        vocabulary=None,

        lowercase=False,
        strip_accents=None,
    )

    ctfidf_model = ClassTfidfTransformer(
        # reduce_frequent_words=True,
        bm25_weighting=True,
    )

    keybert_inspired = KeyBERTInspired(
        top_n_words=top_n_words,
        nr_repr_docs=5,         
        nr_samples=500,       
        nr_candidate_words=100,      
        random_state=42,  
    )

    part_of_speech = PartOfSpeech(
        model="en_core_web_sm",
        top_n_words=top_n_words,
        pos_patterns=[
            # 3-gram patterns (common and meaningful for academic topics)
            [{"POS": "ADJ"}, {"POS": "ADJ"}, {"POS": "NOUN"}],    # e.g., "deep neural network"
            [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN"}],   # e.g., "convolutional neural network"
            [{"POS": "NOUN"}, {"POS": "NOUN"}, {"POS": "NOUN"}],  # e.g., "support vector machine"
            [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],   # e.g., "learning of representations"
            # 2-gram patterns
            [{"POS": "ADJ"}, {"POS": "NOUN"}],                    # e.g., "artificial intelligence"
            [{"POS": "NOUN"}, {"POS": "NOUN"}],                   # e.g., "feature extraction"
            # 1-gram patterns
            [{"POS": "NOUN"}],                                    # e.g., "algorithm"
            [{"POS": "PROPN"}],                                   # e.g., "BERT"
            [{"POS": "ADJ"}],                                     # e.g., "unsupervised"
        ]
    )

    maximal_marginal_relevance = MaximalMarginalRelevance(
        diversity=0.7,
        top_n_words=top_n_words
    )
    representation_models = [keybert_inspired, part_of_speech, maximal_marginal_relevance]

    # # UMAPパラメータをデータセットサイズに応じてproportionで自動調整
    num_texts = len(texts)
    # n_neighborsをデータセットサイズ(num_texts)に応じてシグモイド関数で自動調整
    # f(x) = 10 + 1 / (1 + exp(-0.00005 * (x - 20000)))
    n_neighbors = int(10 + 1 / (1 + np.exp(-0.00005 * (num_texts - 20000))))
    n_components = 5

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric='cosine',
        low_memory=False,
        min_dist=0.0,  
        spread=1.0,
        random_state=42
    )

    # # データセットサイズに応じてmin_cluster_size, min_samplesをproportionで自動調整
    num_texts = len(texts)
    min_cluster_size = int(np.sqrt(num_texts))
    min_samples = int(min_cluster_size // 1.5)

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        prediction_data=True
    )

    model = BERTopic(
        #n_gram_range=(1, 3),
        # min_topic_size=min_df,
        # nr_topics="auto", 
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        representation_model=representation_models,
        embedding_model=embedding_model,
        calculate_probabilities=True,
        verbose=True
    )

    # 学習
    torch.cuda.empty_cache()
    model.fit(texts, embeddings=text_embeddings)

    # 保存
    model.save(f"./models/{category}", serialization="safetensors", save_ctfidf=True)

    # save representative docs with pickle
    representative_docs = model.get_representative_docs()
    with open(f"./models/{category}/representative_docs.pkl", "wb") as f:
        pickle.dump(representative_docs, f)
