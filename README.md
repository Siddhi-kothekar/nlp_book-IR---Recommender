## NLP-based Information Retrieval and Recommendation System (Amazon Reviews)

This project builds an end-to-end pipeline to retrieve and recommend Amazon products using user reviews and sentiments.

### Features
- Information Retrieval over reviews using Sentence-Transformer embeddings and FAISS.
- Sentiment analysis with VADER; optional transformer sentiment ready.
- Recommendation scoring that weights relevance by sentiment and average ratings.
- Streamlit UI for search and recommendations.
- CLI to build the vector index from a CSV of Amazon reviews.

### Project Structure
- `src/`: Core modules (data, preprocessing, sentiment, embeddings, indexer, recommender)
- `app/`: Streamlit application
- `scripts/`: CLI scripts to build index and run the app
- `artifacts/`: Saved indices and metadata
- `data/`: Your input CSV (sample provided)
- `config.yaml`: Configuration for models and paths

### Dataset
Provide a CSV with at least these columns (sample schema):
- `asin`: Product ID
- `title`: Product title/name
- `reviewText`: Free-text user review
- `overall`: Numeric rating (1-5)

A small `data/sample_amazon_reviews.csv` is included as a template.

### Quickstart
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Build the index from your CSV (update `config.yaml` if needed):
```bash
python scripts/build_index.py --data data/sample_amazon_reviews.csv --artifacts_dir artifacts
```
4. Run the app:
```bash
streamlit run app/streamlit_app.py
```

### Configuration (`config.yaml`)
- `model_name`: Sentence-Transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `artifacts_dir`: Directory for index and metadata files
- `top_k`: Default number of retrieved reviews
- `recommendations`: Number of products to recommend

### Notes
- For large datasets, the builder script streams batches to control memory.
- FAISS uses inner product with normalized embeddings for cosine similarity.

### License
MIT
