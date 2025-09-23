from typing import List
from datetime import date, timedelta

import arxiv
from common.repository.model import ArxivPaperModel

class ArxivApi:
    def __init__(self, page_size: int = 100, delay_seconds: float = 3.0, num_retries: int = 3):
        self.client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries
        )

    def search_by_query(
        self,
        query: str,
        max_results: int = 50,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate
    ) -> List[ArxivPaperModel]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )

        results = []
        generator = self.client.results(search)

        while True:
            try:
                paper = next(generator)
                results.append(self._to_model(paper))
            except arxiv.UnexpectedEmptyPageError:
                print(f"[WARN] Empty page encountered for query={query}, skipping...")
                continue  # そのまま次を試す
            except StopIteration:
                break  # 全て取り終わり
            except Exception as e:
                print(f"[ERROR] Unexpected error for query={query}: {e}, skipping...")
                continue

        return results

    def fetch_by_day(
        self,
        target_date: date,
        max_results: int = 50
    ) -> List[ArxivPaperModel]:
        """
        日付で論文を取得
        """
        next_day_date = target_date + timedelta(days=1)

        day_str = target_date.strftime("%Y%m%d")
        next_day_str = next_day_date.strftime("%Y%m%d")

        query = f"submittedDate:[{day_str} TO {next_day_str}]"
        return self.search_by_query(query, max_results)

    def fetch_by_id(
        self,
        id_list: List[str] 
    ) -> List[ArxivPaperModel]:
        """
        idで論文を取得
        """
        search = arxiv.Search(
            id_list=id_list
        )

        results = []
        generator = self.client.results(search)

        while True:
            try:
                paper = next(generator)
                results.append(self._to_model(paper))
            except arxiv.UnexpectedEmptyPageError:
                print(f"[WARN] Empty page encountered for id_list={id_list}, skipping...")
                continue  # そのまま次を試す
            except StopIteration:
                break  # 全て取り終わり
            except Exception as e:
                print(f"[ERROR] Unexpected error for query={id_list}: {e}, skipping...")
                continue

        return results


    def _to_model(self, paper: arxiv.Result) -> ArxivPaperModel:
        return ArxivPaperModel(
            arxiv_id=paper.entry_id.split("abs/")[-1],
            title=paper.title,
            summary=paper.summary,
            authors=[a.name for a in paper.authors],
            categories=paper.categories,
            published=paper.published,
            updated=paper.updated,
            pdf_url=paper.pdf_url,
            entry_id=paper.entry_id,
        )
