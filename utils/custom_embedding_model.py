import numpy as np
import torch
from torch import nn

from transformers import AutoTokenizer
from adapters import AutoAdapterModel

from bertopic.backend import BaseEmbedder

from common.domain.dto import Paper

class CustomEmbeddingModel(BaseEmbedder):
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, device: str = "cuda"):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def embed(self, documents, verbose=False):
        # If documents is a numpy array, convert to list of str
        if isinstance(documents, np.ndarray):
            # If it's an array of strings, flatten and convert to list
            if documents.dtype.type is np.str_ or documents.dtype.type is np.object_:
                documents = documents.tolist()
            else:
                # Try to convert to list of str
                documents = [str(d) for d in documents.flatten()]
        elif not isinstance(documents, (list, tuple)):
            # If it's a single string or something else, wrap in list
            documents = [documents]

        inputs = self.tokenizer(
            documents, 
            padding=True, 
            truncation=True,
            return_tensors="pt", 
            return_token_type_ids=False, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def get_input_text(self, paper: Paper) -> str:
        return f"{paper.title}{self.tokenizer.sep_token}{paper.abstract}"

def get_custom_embedding_model(device: str = "cuda") -> CustomEmbeddingModel:
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2",
        source="hf",
        load_as="proximity",
        set_active=True
    )

    return CustomEmbeddingModel(model, tokenizer, device)