import os, pickle

from datetime import date
from typing import List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from common.domain.service import PaperService
from common.repository.dao import PaperDAO
from common.domain.dto import Paper

def fetch_papers(category: str = "", paper_path: str = "", from_date: date = None, to_date: date = None) -> List[Paper]:
    if os.path.exists(paper_path):
        with open(paper_path, "rb") as f:
            papers = pickle.load(f)
        return papers

    engine = create_engine("mysql+pymysql://papertrends:papertrends@localhost:3306/papertrends")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    paper_dao = PaperDAO(session)
    paper_service = PaperService(paper_dao)

    papers = paper_service.get_papers_by_category(category, from_date=from_date, to_date=to_date)

    session.close()

    with open(paper_path, "wb") as f:
        pickle.dump(papers, f)

    return papers