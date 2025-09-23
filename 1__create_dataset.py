import sys
from pathlib import Path
from datetime import date, timedelta

from tqdm import tqdm

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(str(Path(__file__).parent))

from common.domain.service import ArxivPaperService, PaperService
from common.repository.api import ArxivApi
from common.repository.dao import PaperDAO

# データベース準備
engine = create_engine("mysql+pymysql://papertrends:papertrends@localhost:3306/papertrends")
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# paper DB service
paper_dao = PaperDAO(session)
paper_service: PaperService = PaperService(paper_dao)

# arXiv API service
arxiv_paper_service = ArxivPaperService(ArxivApi(page_size = 500, delay_seconds = 3.00))

# 取得期間
start_date = date(2017, 9, 12)
end_date = date(2025, 9, 17)

total_days = (end_date - start_date).days + 1

# ループして日ごとに処理
pbar = tqdm(range(total_days), desc="Processing days")
for offset in pbar:

    # 処理中のdateを計算
    current_date = start_date + timedelta(days=offset)

    # tqdm のバー横に日付を表示
    pbar.set_postfix({"date": current_date})

    # 論文取得
    arxiv_papers = arxiv_paper_service.get_from_api_by_date(current_date, limit=None)
    
    # ArxivPaper -> Paper に変換
    papers = [arxiv_paper.to_paper() for arxiv_paper in arxiv_papers]
    
    # 登録
    paper_service.register_papers(papers)
    session.commit()

session.close()
    

