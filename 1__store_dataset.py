# store dataset from database to file for efficient processing

import os
from pathlib import Path
from typing import List
from papertrends_dataset_lib.domain import Paper, PaperService
from papertrends_dataset_lib.domain.service import PaperDAO
from papertrends_dataset_lib.utils import ConfigLoader
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import date
import pickle
import numpy as np
from tqdm import tqdm

from utils.custom_embedder import Specter2Embedder

# ============================================================================
# Hyperparameters
# ============================================================================

FROM_DATE = date(2025, 1, 1)

# ============================================================================
# Singleton
# ============================================================================

CONFIG_LOADER = ConfigLoader(Path(__file__).parent / "config")

engine = create_engine(CONFIG_LOADER.get_config_value("database", "url"))
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

PAPER_SERVICE = PaperService(PaperDAO(session))


def save_dataset(category_name: str, subcategory: str, papers: List[Paper]):
    """Save dataset files for a category/subcategory."""
    base_dir = f"./dataset/{category_name}/{subcategory}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Extract data
    ids = [paper.id for paper in papers]
    arxiv_ids = [paper.arxiv_id for paper in papers]
    titles = [paper.title for paper in papers]
    abstracts = [paper.abstract for paper in papers]
    published_dates = [paper.published for paper in papers]
    embeddings = PAPER_SERVICE.get_embeddings_as_ndarray(papers, embedding_type="specter2")
    
    # Save files
    files_to_save = {
        'ids.pkl': ids,
        'arxiv_ids.pkl': arxiv_ids,
        'titles.pkl': titles,
        'abstracts.pkl': abstracts,
        'published_dates.pkl': published_dates,
        'embeddings.pkl': embeddings
    }
    
    for filename, data in files_to_save.items():
        with open(f"{base_dir}/{filename}", "wb") as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    categories = CONFIG_LOADER.load_yaml("categories.yaml")
    
    # Calculate total number of subcategories for progress tracking
    total_subcategories = sum(len(category_items) for category_items in categories.values())
    
    with tqdm(total=total_subcategories, desc="Processing categories", unit="subcategory") as pbar:
        for category_name, category_items in categories.items():
            os.makedirs(f"./dataset/{category_name}", exist_ok=True)

            for subcategory in category_items:
                pbar.set_description(f"Processing {category_name}/{subcategory}")
                
                # Fetch and save dataset
                papers = PAPER_SERVICE.list(
                    category=subcategory, 
                    from_date=FROM_DATE, 
                    include_embeddings=True, 
                    embedding_type="specter2"
                )
                save_dataset(category_name, subcategory, papers)
                
                pbar.update(1)


