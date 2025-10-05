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
    """Load pre-computed SPECTER2 text embeddings."""
    filepath = f"./preprocessed/{category}/text_embeddings.npy"
    try:
        with open(filepath, "rb") as f:
            return np.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Text embeddings not found at {filepath}")

def create_model(params: Hyperparameters) -> BERTopic:
    vectorizer_model = CountVectorizer(
        stop_words="english",
        analyzer="word",
        ngram_range=tuple(params.ngram_range),
        min_df=params.min_df,
        max_df=params.max_df,
        lowercase=False,
        strip_accents=None
    )
    
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
    
    umap_model = UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        metric=params.umap_metric,
        random_state=42,
        low_memory=False
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric=params.hdbscan_metric,
        prediction_data=True
    )

    # representations(topic名、代表単語が変わる)
    top_n_words = params.top_n_words
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
        top_n_words=top_n_words,
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

    models_path = f"./models/{category}"
    if(os.path.exists(models_path)):
        # すでに訓練済みの場合はスキップ
        return

    # 前処理済データを取得
    papers = load_papers(category)   
    text_embeddings = load_text_embeddings(category)
    texts = [EMBEDDING_MODEL.get_input_text(paper) for paper in papers]
    del papers
    gc.collect()

    # パラメータを取得
    param_path = f"./params/{category}/best_params.json"
    with open(param_path, "r") as f:
        best_params = json.load(f)

    hyperparameters = Hyperparameters(**best_params)

    # モデルを訓練
    model = create_model(hyperparameters)
    model.fit(texts, embeddings=text_embeddings)

    model.save(models_path, serialization="safetensors", save_ctfidf=True)

    # save representative docs with pickle
    representative_docs = model.get_representative_docs()
    with open(f"{models_path}/representative_docs.pkl", "wb") as f:
        pickle.dump(representative_docs, f)

if __name__ == "__main__":
    category = "cs.AR"
    process_one_category(category)

