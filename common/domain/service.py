from typing import List
from datetime import date, datetime

from common.domain.dto import ArxivPaper, Paper, Keyword
from common.repository.dao import PaperDAO, KeywordDAO
from common.repository.entity import PaperEntity, PaperCategoryEntity, KeywordEntity
from common.repository.api import ArxivApi
from common.repository.model import ArxivPaperModel

class PaperService:
    def __init__(self, dao: PaperDAO):
        self.dao = dao

    def register_papers(self, papers: List[Paper]):
        """
        Convert Paper DTO to PaperEntity and bulk save
        """
        paper_entities = []
        category_entities = []

        for p in papers:
            paper_entity = PaperEntity(
                arxiv_id=p.arxiv_id,
                title=p.title,
                abstract=p.abstract,
                published=p.published,
            )

            paper_entities.append(paper_entity)

            for c in set([cat.category.lower() for cat in p.categories]):
                category_entities.append(
                    PaperCategoryEntity(
                        paper=paper_entity,  # paper_id is set via relationship
                        category=c
                    )
                )

        self.dao.save_bulk(paper_entities, category_entities)

    def get_papers_by_category(
        self, 
        category: str, 
        from_date: date = None, 
        to_date: date = None,
        start_idx: int = None,
        end_idx: int = None
    ) -> List[Paper]:
        """
        Convert PaperEntity to Paper DTO
        Period can be specified with from_datetime, to_datetime
        Range can also be specified with start_idx, end_idx
        """
        limit = None
        offset = None
        if start_idx is not None and end_idx is not None:
            offset = start_idx
            limit = end_idx - start_idx
        elif start_idx is not None:
            offset = start_idx
        elif end_idx is not None:
            limit = end_idx

        models = self.dao.list_by_category(
            category=category, 
            from_date=from_date, 
            to_date=to_date,
            limit=limit,
            offset=offset
        )
        
        papers = []
        for m in models:
            # PaperEntity.categories is a list of PaperCategoryEntity
            categories = [
                c.category for c in m.categories
            ]
            papers.append(
                Paper(
                    arxiv_id=m.arxiv_id,
                    published=m.published,
                    title=m.title,
                    abstract=m.abstract,
                    categories=categories
                )
            )
        return papers

class KeywordService:
    def __init__(self, dao: KeywordDAO):
        self.dao = dao

    def register_keywords(self, keywords: List[Keyword]):
        """
        Convert Keyword DTO to KeywordEntity and bulk insert
        Supports daily aggregation (abstract_count, title_count)
        """
        models = [
            KeywordEntity(
                token=k.token,
                category=k.category,
                year=k.year,
                month=k.month,
                day=k.day,
                abstract_count=k.abstract_count,
                title_count=k.title_count or 0
            )
            for k in keywords
        ]
        self.dao.save_bulk(models)

    def get_keywords(self, category: str, year: int, month: int, day: int) -> List[Keyword]:
        """
        Convert KeywordEntity to Keyword DTO
        Daily specification is possible (if not specified, all records are retrieved by month)
        """
        models = self.dao.list_by_category_and_period(category, year, month, day)

        return [
            Keyword(
                token=m.token,
                category=m.category,
                year=m.year,
                month=m.month,
                day=m.day,
                count=m.count,
            )
            for m in models
        ]

class ArxivPaperService:
    def __init__(self, api: ArxivApi):
        self.api = api

    def get_from_api_by_date(self, target_date: date, limit: int = 10) -> List[ArxivPaper]:
        """
        Fetch from arXiv
        """

        papers = self.api.fetch_by_day(
            target_date=target_date,
            max_results=limit
        )

        return [
            ArxivPaper(
                arxiv_id=p.arxiv_id,
                title=p.title,
                summary=p.summary,
                authors=p.authors, 
                categories=p.categories if isinstance(p.categories, list) else [p.categories],
                published_date=p.published,
                updated_date=p.updated,
                pdf_url=getattr(p, "pdf_url", None),
                entry_id=getattr(p, "entry_id", None)
            )
            for p in papers
        ]

    def get_from_api_by_arxiv_id(self, arxiv_id_list: List[str]) -> Paper:
        """
        Fetch from arXiv
        """

        papers = self.api.fetch_by_id(
            id_list = arxiv_id_list
        )

        return [
            ArxivPaper(
                arxiv_id=p.arxiv_id,
                title=p.title,
                summary=p.summary,
                authors=p.authors, 
                categories=p.categories if isinstance(p.categories, list) else [p.categories],
                published_date=p.published,
                updated_date=p.updated,
                pdf_url=getattr(p, "pdf_url", None),
                entry_id=getattr(p, "entry_id", None)
            )
            for p in papers
        ]