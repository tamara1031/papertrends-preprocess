from sqlalchemy import Column, BigInteger, String, Boolean, Date, ForeignKey, Index, UniqueConstraint, Integer
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class PaperEntity(Base):
    __tablename__ = "papers"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    arxiv_id = Column(String(32), nullable=False, unique=True)
    title = Column(String(255), nullable=False)
    abstract = Column(String, nullable=True)
    published = Column(Date, nullable=False)

    categories = relationship("PaperCategoryEntity", back_populates="paper")


class PaperCategoryEntity(Base):
    __tablename__ = "paper_categories"
    __table_args__ = (
        UniqueConstraint('paper_id', 'category', name='unique_paper_category'),
        Index("idx_category", "category"),
    )

    paper_id = Column(BigInteger, ForeignKey("papers.id"), primary_key=True)
    category = Column(String(50), primary_key=True)

    paper = relationship("PaperEntity", back_populates="categories")


class KeywordEntity(Base):
    __tablename__ = "keywords"
    __table_args__ = (
        UniqueConstraint('token', 'category', 'published', name='unique_token_category_day'),
        Index('idx_keywords_category_published', 'category', 'published'),  # カテゴリ・日付検索用
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    token = Column(String(255), nullable=False)
    category = Column(String(255), nullable=False)
    published = Column(Date, nullable=False)
    count = Column(Integer, nullable=False)
