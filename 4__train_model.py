from typing import List, Tuple, Any, Optional, Dict, Union
from dataclasses import dataclass

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

EMBEDDING_MODEL = get_custom_embedding_model()

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

def get_papers(category: str) -> List[Paper]:
    with open(f"./preprocessed/{category}/papers.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def get_text_embeddings(category: str) -> np.ndarray:
    with open(f"./preprocessed/{category}/text_embeddings.npy", "rb") as f:
        embeddings = np.load(f)
    return embeddings

def create_model(params: Hyperparameters) -> BERTopic:
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=params.ngram_range,
        min_df=params.min_df,
        max_df=params.max_df,
        max_features=None,
        vocabulary=None,

        lowercase=params.lowercase,
        strip_accents=params.strip_accents,
    )
    ctfidf_model = ClassTfidfTransformer(
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

    # representations(topic名、代表単語が変わる)
    top_n_words = 10
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

    model = BERTopic(
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        representation_model=representation_models,
        embedding_model=EMBEDDING_MODEL,
        calculate_probabilities=True,
        verbose=True
    )

    return model

def process_one_category(category: str):

    # 前処理済データを取得
    papers = get_papers(category)
    text_embeddings = get_text_embeddings(category)
    texts = [EMBEDDING_MODEL.get_input_text(paper) for paper in papers]
    del papers
    gc.collect()

    # パラメータを取得
    param_dir = f"./models/{category}/best_params.json"
    with open(param_dir, "r") as f:
        best_params = json.load(f)

    hyperparameters = Hyperparameters(**best_params)

    # モデルを訓練
    model = create_model(hyperparameters)
    model.fit(texts, embeddings=text_embeddings)

    model.save(f"./models/{category}", serialization="safetensors", save_ctfidf=True)

    # save representative docs with pickle
    representative_docs = model.get_representative_docs()
    with open(f"./models/{category}/representative_docs.pkl", "wb") as f:
        pickle.dump(representative_docs, f)

if __name__ == "__main__":
    category = "physics.geo-ph"
    process_one_category(category)

