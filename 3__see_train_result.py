import pickle
from bertopic import BERTopic
from common.utils import get_custom_embedding_model

EMBEDDING_MODEL = get_custom_embedding_model()

def load_model(category: str):
    model = BERTopic.load(f"./models/{category}", embedding_model=EMBEDDING_MODEL)
    return model

def load_papers(category: str):
    with open(f"./preprocessed/{category}/papers.pkl", "rb") as f:
        papers = pickle.load(f)
    return papers

if __name__ == "__main__":
    category = "cs.AR"
    model = load_model(category)
    vis = model.visualize_topics()
    vis.show()

