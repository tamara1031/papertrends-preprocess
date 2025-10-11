import pickle
from bertopic import BERTopic
from utils.custom_embedder import Specter2Embedder

# Initialize embedding model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = Specter2Embedder(device=device)

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

