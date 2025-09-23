-- 論文メタデータテーブル（日単位集計用）
CREATE TABLE papers (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    arxiv_id VARCHAR(32) NOT NULL UNIQUE,
    title VARCHAR(512) NOT NULL,
    abstract TEXT,
    published DATE NOT NULL
);

-- 論文カテゴリテーブル
CREATE TABLE paper_categories (
    paper_id BIGINT NOT NULL,
    category VARCHAR(512) NOT NULL COLLATE utf8mb4_general_ci,
    PRIMARY KEY (paper_id, category),
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    INDEX idx_category (category)
);

-- 時系列集計テーブル（日単位集計用、タイトル出現数対応）
CREATE TABLE IF NOT EXISTS keywords (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    token VARCHAR(255) NOT NULL,
    category VARCHAR(255) NOT NULL,
    published DATE NOT NULL,
    count INT NOT NULL, -- 出現数
    UNIQUE KEY unique_token_category_day (token, category, published),
    INDEX idx_keywords_category_published (category, published)  -- カテゴリ・日付検索用
);
