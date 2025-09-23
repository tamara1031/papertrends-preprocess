from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, date
from common.repository.entity import PaperEntity, PaperCategoryEntity, KeywordEntity

class PaperDAO:
    def __init__(self, session: Session):
        self.session = session

    def save_bulk(self, papers: List[PaperEntity], categories: List[PaperCategoryEntity]):
        """
        Bulk registration
        Register PaperEntity and PaperCategoryEntity together
        """
        # Collect arxiv_ids
        arxiv_ids = [p.arxiv_id for p in papers]

        # Get existing Papers by arxiv_id -> PaperEntity
        existing_papers = {
            r.arxiv_id: r
            for r in self.session.query(PaperEntity)
            .filter(PaperEntity.arxiv_id.in_(arxiv_ids))
            .all()
        }

        # Only new PaperEntity
        new_papers = [p for p in papers if p.arxiv_id not in existing_papers]

        if new_papers:
            self.session.add_all(new_papers)
            self.session.flush()

    def list_by_category(
        self,
        category: str = None,
        from_date: date = None,
        to_date: date = None,
        limit: int = None,
        offset: int = None
    ) -> List[PaperEntity]:
        """
        Bulk retrieve PaperEntity belonging to specified category
        limit and offset can also be specified
        from_date, to_date are specified as date type
        """
        query = (
            self.session.query(PaperEntity)
            .join(PaperCategoryEntity, PaperEntity.id == PaperCategoryEntity.paper_id)
        )
        if category is not None:
            query = query.filter(PaperCategoryEntity.category == category)
        if from_date is not None:
            query = query.filter(PaperEntity.published >= from_date)
        if to_date is not None:
            query = query.filter(PaperEntity.published <= to_date)
        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        return query.all()

class KeywordDAO:
    def __init__(self, session: Session):
        self.session = session

    def list_by_category_and_period(self, category: str, year: int, month: int) -> List[KeywordEntity]:
        return self.session.query(KeywordEntity).filter_by(
            category=category,
            year=year,
            month=month
        ).all()

    def upsert_counts(self, entity: KeywordEntity):
        """
        Upsert keywords and update count
        """
        stmt = insert(KeywordEntity).values(
            token=entity.token,
            category=entity.category,
            year=entity.year,
            month=entity.month,
            day=entity.day,
            title_count=entity.title_count,
            abstract_count=entity.abstract_count
        )
        stmt = stmt.on_duplicate_key_update(
            title_count=KeywordEntity.title_count + entity.title_count,
            abstract_count=KeywordEntity.abstract_count + entity.abstract_count
        )
        self.session.execute(stmt)

