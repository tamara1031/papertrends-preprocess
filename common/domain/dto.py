from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List

@dataclass
class PaperCategory:
    category: str

@dataclass
class Paper:
    arxiv_id: str
    published: date
    title: str
    abstract: str
    categories: List[PaperCategory]  # Paper にカテゴリを持たせる

@dataclass
class Keyword:
    token: str
    category: str
    published: date
    count: int

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    categories: List[str]
    published_date: datetime
    updated_date: Optional[datetime] = None
    pdf_url: Optional[str] = None
    entry_id: Optional[str] = None

    def to_paper(self) -> Paper:
        """Paper DTO に変換、カテゴリを埋め込む"""
        return Paper(
            arxiv_id=self.arxiv_id,
            published=self.published_date.date(),
            title=self.title,
            abstract=self.summary,
            categories=[PaperCategory(category=c) for c in self.categories]
        )