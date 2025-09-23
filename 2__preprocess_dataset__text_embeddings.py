from typing import List

from datetime import date
import os, gc

from tqdm import tqdm
import numpy as np
import torch

from common.utils import (
    get_category_codes,
    get_custom_embedding_model,
    CustomEmbeddingModel,
    fetch_papers,
)

# PARAMS
BATCH_SIZE = 128
FROM_DATE = date(2020, 1, 1)
CATEGORIES = ["cs.IR"]
#CATEGORIES = get_category_codes()

def encode_texts(model: CustomEmbeddingModel, texts: List[str], memmap_path: str):

    # すでに処理済みの場合はスキップ
    if os.path.exists(memmap_path):
        return

    num_texts = len(texts)
    num_batches = (num_texts + BATCH_SIZE - 1) // BATCH_SIZE
    memmap_file = None

    for i in tqdm(range(num_batches), leave=False):
        batch_texts = texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_embeddings = model.embed(batch_texts)

        start_idx = i * BATCH_SIZE

        if memmap_file is None:
            total_shape = (num_texts, batch_embeddings.shape[1])
            memmap_file = np.lib.format.open_memmap(
                memmap_path, mode='w+', dtype=batch_embeddings.dtype, shape=total_shape
            )
        end_idx = start_idx + batch_embeddings.shape[0]
        memmap_file[start_idx:end_idx, :] = batch_embeddings
        memmap_file.flush()

        # clean up batch
        del batch_embeddings, batch_texts
        gc.collect() 

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_custom_embedding_model(device)

with tqdm(CATEGORIES, desc="Processing categories") as pbar:
    for category in pbar:
        paper_path = f"./preprocessed/{category}/papers.pkl"
        mmap_path = f"./preprocessed/{category}/text_embeddings.npy"

        # すでに処理済みの場合はスキップ
        if os.path.exists(mmap_path) and os.path.exists(paper_path):
            continue

        # ディレクトリを作成
        os.makedirs(f"./preprocessed/{category}", exist_ok=True)

        # 以下処理
        pbar.set_postfix_str(f"Fetching papers for {category}")
        papers = fetch_papers(category, paper_path, from_date=FROM_DATE)

        pbar.set_postfix_str(f"Converting papers to texts for {category}")
        texts = [
            model.get_input_text(paper)
            for paper in papers
        ]
        del papers  # Free memory after conversion
        gc.collect()
        
        pbar.set_postfix_str(f"BatchEncoding Texts for {category}")
        torch.cuda.empty_cache() # 実行前にキャッシュをクリア
        encode_texts(model, texts, mmap_path)
        del texts # Free memory after encoding
        gc.collect()

        pbar.set_postfix_str(f"Done with {category}")