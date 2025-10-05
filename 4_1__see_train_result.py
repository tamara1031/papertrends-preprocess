import pickle
from bertopic import BERTopic
from common.utils import get_custom_embedding_model

def load_model(category: str):
    embedding_model = get_custom_embedding_model()
    model = BERTopic.load(f"./models/{category}", embedding_model=embedding_model)
    return model

def load_papers(category: str):
    with open(f"./preprocessed/{category}/papers.pkl", "rb") as f:
        papers = pickle.load(f)
    return papers

