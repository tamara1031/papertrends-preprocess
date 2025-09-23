-- Paper metadata table (for daily aggregation)
CREATE TABLE papers (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    arxiv_id VARCHAR(32) NOT NULL UNIQUE,
    title VARCHAR(512) NOT NULL,
    abstract TEXT,
    published DATE NOT NULL
);

-- Paper category table
CREATE TABLE paper_categories (
    paper_id BIGINT NOT NULL,
    category VARCHAR(512) NOT NULL COLLATE utf8mb4_general_ci,
    PRIMARY KEY (paper_id, category),
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    INDEX idx_category (category)
);
