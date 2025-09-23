from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class ArxivPaperModel:
    """
    arXiv API から取得した論文データを表すモデル。
    Service 層で PaperEntity に変換して保存する。
    """
    arxiv_id: str                 # arXiv の固有ID（例: 2309.12345v1）
    title: str
    summary: str                  # abstract 
    authors: List[str]
    categories: List[str]
    published: datetime
    updated: datetime
    pdf_url: Optional[str] = None
    entry_id: Optional[str] = None