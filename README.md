# papertrends-preprocess

arXiv論文のトピックモデリングと可視化データ生成のための前処理パイプラインです。

## 概要

このプロジェクトは、arXivから論文データを取得し、BERTopicを使用してトピックモデリングを行い、可視化用のデータを生成する一連の処理を提供します。

## セットアップ

### 前提条件

- Python 3.11（PyTorch互換バージョン）
- MySQL 8.0
- CUDA対応GPU（推奨）

### インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# spaCyの英語モデルをダウンロード
python -m spacy download en_core_web_sm
```

### データベースセットアップ

```bash
# Docker Composeを使用してMySQLを起動
docker-compose up -d

# データベースが起動するまで待機（約30秒）
```

## 使用方法

### 1. データセット作成

arXivから論文データを取得してデータベースに保存します。

```bash
python 1__create_dataset.py
```

**設定項目:**
- `start_date`: 取得開始日（デフォルト: 2017-09-12）
- `end_date`: 取得終了日（デフォルト: 2025-09-17）
- カテゴリ: 全arXivカテゴリから取得

### 2. テキスト埋め込み生成

論文のタイトルとアブストラクトからテキスト埋め込みを生成します。

```bash
python 2__preprocess_dataset__text_embeddings.py
```

**設定項目:**
- `BATCH_SIZE`: バッチサイズ（デフォルト: 128）
- `FROM_DATE`: 処理対象の開始日（デフォルト: 2020-01-01）
- `CATEGORIES`: 処理対象カテゴリ（デフォルト: ["cs.IR"]）

**出力:**
- `./preprocessed/{category}/papers.pkl`: 論文データ
- `./preprocessed/{category}/text_embeddings.npy`: テキスト埋め込み

### 3. トピックモデル学習

BERTopicを使用してトピックモデルを学習します。

```bash
python 3__train_model.py
```

**設定項目:**
- `categories`: 学習対象カテゴリ（デフォルト: ["cs.IR"]）

**出力:**
- `./models/{category}/`: 学習済みモデル
- `./models/{category}/representative_docs.pkl`: 代表文書

### 4. 可視化データ生成

トピックモデルから可視化用のJSONデータを生成します。

```bash
python 4__generate_visualizization_data.py
```

**出力:**
- `./visualizations/{category}.json`: 可視化データ

**生成されるデータ:**
- トピック情報（名前、キーワード、論文数）
- トピック間相関行列
- 時系列データ（月ごとのトピック分布）
- 各トピックの代表論文

## ディレクトリ構造

```
papertrends-preprocess/
├── 1__create_dataset.py              # arXivデータ取得
├── 2__preprocess_dataset__text_embeddings.py  # テキスト埋め込み生成
├── 3__train_model.py                 # トピックモデル学習
├── 4__generate_visualizization_data.py  # 可視化データ生成
├── common/                           # 共通モジュール
├── config/                           # 設定ファイル
├── models/                           # 学習済みモデル
├── preprocessed/                     # 前処理済みデータ
├── visualizations/                   # 可視化データ
├── mysql/                            # MySQLデータ
└── docker-compose.yml               # Docker設定
```

## 設定のカスタマイズ

各スクリプト内のパラメータを変更することで、処理対象の期間やカテゴリを調整できます。

### 主要パラメータ

- **データ取得期間**: `1__create_dataset.py`の`start_date`、`end_date`
- **処理対象カテゴリ**: 各スクリプトの`CATEGORIES`変数
- **バッチサイズ**: `2__preprocess_dataset__text_embeddings.py`の`BATCH_SIZE`
- **クラスタリングパラメータ**: `3__train_model.py`のHDBSCAN設定

## 注意事項

- GPUメモリが不足する場合は、`BATCH_SIZE`を小さくしてください
- 大量のデータを処理する場合は、十分なディスク容量を確保してください

## ライセンス

このプロジェクトのライセンスについては、[LICENSE](LICENSE)ファイルを参照してください。
