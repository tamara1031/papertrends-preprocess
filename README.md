# papertrends-preprocess

A preprocessing pipeline for arXiv paper topic modeling and visualization data generation.

## Overview

This project provides a series of processes that fetch paper data from arXiv, perform topic modeling using BERTopic, and generate visualization data.

## Setup

### Prerequisites

- Python 3.11 (PyTorch compatible version)
- MySQL 8.0
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Database Setup

```bash
# Start MySQL using Docker Compose
docker-compose up -d

# Wait for database to start (approximately 30 seconds)
```

## Usage

### 1. Dataset Creation

Fetch paper data from arXiv and save to database.

```bash
python 1__create_dataset.py
```

**Configuration:**
- `start_date`: Start date for data collection (default: 2017-09-12)
- `end_date`: End date for data collection (default: 2025-09-17)
- Categories: All arXiv categories are collected

### 2. Text Embedding Generation

Generate text embeddings from paper titles and abstracts.

```bash
python 2__preprocess_dataset__text_embeddings.py
```

**Configuration:**
- `BATCH_SIZE`: Batch size (default: 128)
- `FROM_DATE`: Start date for processing (default: 2020-01-01)
- `CATEGORIES`: Target categories for processing (default: ["cs.IR"])

**Output:**
- `./preprocessed/{category}/papers.pkl`: Paper data
- `./preprocessed/{category}/text_embeddings.npy`: Text embeddings

### 3. Topic Model Training

Train topic models using BERTopic.

```bash
python 3__train_model.py
```

**Configuration:**
- `categories`: Target categories for training (default: ["cs.IR"])

**Output:**
- `./models/{category}/`: Trained models
- `./models/{category}/representative_docs.pkl`: Representative documents

### 4. Visualization Data Generation

Generate JSON data for visualization from topic models.

```bash
python 4__generate_visualizization_data.py
```

**Output:**
- `./visualizations/{category}.json`: Visualization data

**Generated Data:**
- Topic information (name, keywords, paper count)
- Topic correlation matrix
- Time series data (monthly topic distribution)
- Representative papers for each topic

## Directory Structure

```
papertrends-preprocess/
├── 1__create_dataset.py              # arXiv data collection
├── 2__preprocess_dataset__text_embeddings.py  # Text embedding generation
├── 3__train_model.py                 # Topic model training
├── 4__generate_visualizization_data.py  # Visualization data generation
├── common/                           # Common modules
├── config/                           # Configuration files
├── models/                           # Trained models
├── preprocessed/                     # Preprocessed data
├── visualizations/                   # Visualization data
├── mysql/                            # MySQL data
└── docker-compose.yml               # Docker configuration
```

## Configuration Customization

You can adjust the processing period and categories by modifying parameters in each script.

### Key Parameters

- **Data collection period**: `start_date`, `end_date` in `1__create_dataset.py`
- **Target categories**: `CATEGORIES` variable in each script
- **Batch size**: `BATCH_SIZE` in `2__preprocess_dataset__text_embeddings.py`
- **Clustering parameters**: HDBSCAN settings in `3__train_model.py`

## Notes

- Reduce `BATCH_SIZE` if GPU memory is insufficient
- Ensure sufficient disk space when processing large amounts of data

## License

Please refer to the [LICENSE](LICENSE) file for this project's license.
