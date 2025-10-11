from bertopic.backend import BaseEmbedder

from papertrends_dataset_lib.domain import Paper
from papertrends_dataset_lib.embedding_models import SPECTER2

class Specter2Embedder(BaseEmbedder):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.model = SPECTER2(device=device)
    
    def embed(self, documents, verbose=False):
        # If documents is a numpy array, convert to list of str
        return self.model.embed(documents)

    def get_input_text(self, title: str, abstract: str) -> str:
        return self.model.get_input_text(title, abstract)